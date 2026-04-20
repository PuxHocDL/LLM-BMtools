import json
import re
from collections import Counter
import copy

class JSONPruner:
    """
    Giảm kích thước file JSON lớn xuống mức tối thiểu.
    Có 2 chiến lược:
    - Lookup: câu hỏi tìm 1 record cụ thể → giữ top_k records match nhất
    - Aggregation: câu hỏi đếm/liệt kê → pre-compute kết quả rồi feed summary
    """

    _AGG_PATTERNS = [
        r'\bhow many\b', r'\bcount\b', r'\bnumber of\b', r'\bprovide the number\b',
        r'\blist the\b', r'\blist all\b', r'\bdifferent .+ available\b',
        r'\bnames of all\b', r'\btypes of\b', r'\ball the\b',
        r'\baccession number of all\b', r'\bgive the accession\b',
    ]

    @staticmethod
    def _extract_query_keywords(query):
        query_words = set(re.findall(r'\w+', query.lower()))
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'in', 'of', 'and', 
                     'for', 'with', 'on', 'at', 'what', 'which', 'how', 'tell', 'me', 'i', 
                     'want', 'know', 'can', 'you', 'give', 'find', 'don', 't', 'include',
                     'provide', 'number', 'list', 'output', 'separated', 'comma'}
        keywords = query_words - stopwords
        return keywords if keywords else query_words

    @staticmethod
    def _score_text(text, keywords):
        text_lower = text.lower()
        return sum(1 for w in keywords if w in text_lower)

    @classmethod
    def _is_aggregation_query(cls, query):
        q_lower = query.lower()
        return any(re.search(p, q_lower) for p in cls._AGG_PATTERNS)

    @classmethod
    def _collect_leaf_records(cls, data, path=""):
        records = []
        if isinstance(data, dict):
            all_scalar = all(not isinstance(v, (dict, list)) for v in data.values())
            if all_scalar and data:
                records.append((path, data))
            else:
                for key, value in data.items():
                    child_path = f"{path}.{key}" if path else key
                    records.extend(cls._collect_leaf_records(value, child_path))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                child_path = f"{path}[{i}]"
                records.extend(cls._collect_leaf_records(item, child_path))
        return records

    @classmethod
    def _build_aggregation_summary(cls, records, query, keywords):
        """
        Pre-compute aggregation results from ALL records and return a compact summary
        that the LLM can use to answer counting/listing questions.
        """
        # Extract all field names from records
        all_fields = set()
        for _, rec in records:
            all_fields.update(rec.keys())
        
        summary_parts = []
        summary_parts.append(f"=== DATA SUMMARY (Total records: {len(records)}) ===\n")
        
        # For each field, compute value counts
        for field in sorted(all_fields):
            values = []
            for _, rec in records:
                if field in rec:
                    values.append(str(rec[field]))
            
            if not values:
                continue
            
            counter = Counter(values)
            unique_count = len(counter)
            
            # Check if this field is relevant to the query
            field_relevant = any(kw in field.lower() for kw in keywords)
            
            if unique_count <= 100 or field_relevant:
                summary_parts.append(f"\n--- Field: {field} (unique values: {unique_count}) ---")
                # Show value counts sorted by frequency
                for val, cnt in counter.most_common(100):
                    summary_parts.append(f"  {val}: {cnt} occurrences")
        
        # Also add year-based breakdown if dates exist
        date_fields = [f for f in all_fields if 'date' in f.lower() or 'period' in f.lower()]
        for df in date_fields:
            year_counter = Counter()
            for _, rec in records:
                if df in rec:
                    year = str(rec[df])[:4]
                    if year.isdigit():
                        year_counter[year] += 1
            if year_counter:
                summary_parts.append(f"\n--- {df} by year ---")
                for year, cnt in sorted(year_counter.items()):
                    summary_parts.append(f"  {year}: {cnt} filings")
        
        # If query asks about specific filter conditions, also add filtered records
        # e.g., "filings filed in year 2016 which are of form type 10-K"
        filtered = cls._try_filter_records(records, query, keywords)
        if filtered is not None:
            summary_parts.append(f"\n--- Filtered records matching query conditions ({len(filtered)} results) ---")
            for _, rec in filtered[:200]:
                summary_parts.append(f"  {json.dumps(rec)}")
        
        return "\n".join(summary_parts)

    @classmethod
    def _try_filter_records(cls, records, query, keywords):
        """Try to extract filter conditions from query and apply them."""
        q_lower = query.lower()
        filtered = list(records)  # start with all
        
        # Extract year filter
        year_match = re.search(r'\byear\s+(\d{4})\b', q_lower)
        if year_match:
            year = year_match.group(1)
            filtered = [(p, r) for p, r in filtered 
                       if any(str(v).startswith(year) for v in r.values())]
        
        # Extract form type filter
        formtype_match = re.search(r'form\s*type\s+(\S+)', q_lower)
        if formtype_match:
            ft = formtype_match.group(1).strip('.,?!')
            filtered = [(p, r) for p, r in filtered
                       if any(str(v).upper() == ft.upper() for v in r.values())]
        
        # Extract name filter
        name_match = re.search(r'(?:name|named)\s+(.+?)(?:\s+and\s+|\s*\?\s*$|\s*\.\s*$|\s*$)', q_lower)
        if name_match:
            name_val = name_match.group(1).strip().rstrip('.,?!')
            # Also try extracting quoted names or specific patterns like "10-K 2017 Report"
            report_match = re.search(r'name\s+([\w\-/]+(?:\s+\w+)*\s+report)', q_lower)
            if report_match:
                name_val = report_match.group(1).strip()
            filtered = [(p, r) for p, r in filtered
                       if any(name_val.lower() in str(v).lower() for v in r.values())]
        
        # Extract specific date conditions (filing date == period, or days between is zero)
        date_match = re.search(r'(?:filing date and period.*?(?:same|equal|match)|days between.*?(?:period|filing).*?(?:zero|0|same))', q_lower)
        if date_match:
            filtered = [(p, r) for p, r in filtered
                       if 'filingDate' in r and 'period' in r and 
                       str(r['filingDate'])[:10] == str(r['period'])[:10]]
        
        # Only return if we actually filtered something
        if len(filtered) < len(records):
            return filtered
        return None

    @classmethod
    def prune(cls, json_str, query, top_k=50):
        try:
            data = json.loads(json_str)
        except Exception:
            return json_str

        keywords = cls._extract_query_keywords(query)
        records = cls._collect_leaf_records(data)
        
        if not records:
            return json_str[:8000]
        
        # Route to different strategy based on query type
        if cls._is_aggregation_query(query):
            return cls._build_aggregation_summary(records, query, keywords)
        
        # === LOOKUP strategy: find specific records ===
        scored = []
        for path, record in records:
            record_text = " ".join(str(k) + " " + str(v) for k, v in record.items())
            full_text = path + " " + record_text
            score = cls._score_text(full_text, keywords)
            scored.append((score, path, record))
        
        scored.sort(key=lambda x: (-x[0], len(x[1])))
        
        top_records = [(p, r) for s, p, r in scored[:top_k] if s > 0]
        
        if not top_records:
            top_records = [(p, r) for _, p, r in scored[:top_k]]
        
        output_records = []
        for path, record in top_records:
            entry = {"_path": path}
            entry.update(record)
            output_records.append(entry)
        
        return json.dumps(output_records, indent=2)


class SemanticToolRetriever:
    """
    Rút gọn danh sách API/Tools truyền vào prompt cho LLM.
    Chỉ giữ lại top_k tools có mô tả liên quan nhất đến câu hỏi.
    """
    @classmethod
    def retrieve(cls, user_query, tools_list, top_k=3):
        if not tools_list or len(tools_list) <= top_k:
            return tools_list
            
        query_words = set(re.findall(r'\w+', user_query.lower()))
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'in', 'of', 'and', 
                     'for', 'with', 'what', 'which', 'how', 'tell', 'me', 'i', 'want', 'know', 
                     'can', 'you', 'need', 'please', 'also', 'get', 'find', 'this', 'that',
                     'not', 'first', 'time', 'try', 'task', 'all', 'previous', 'trails', 'failed',
                     'before', 'based', 'on', 'your'}
        query_words = query_words - stopwords

        scored_tools = []
        for tool in tools_list:
            tool_name = str(tool.get('name', '')).replace('_', ' ').lower()
            tool_desc = str(tool.get('description', '')).lower()
            
            # Also score on parameter names and descriptions
            params_text = ""
            params = tool.get('parameters', {})
            if isinstance(params, dict):
                props = params.get('properties', {})
                for pname, pinfo in props.items():
                    params_text += " " + pname.replace('_', ' ')
                    if isinstance(pinfo, dict):
                        params_text += " " + str(pinfo.get('description', ''))
            params_text = params_text.lower()
            
            text_to_search = tool_name + " " + tool_desc + " " + params_text
            
            score = 0
            for w in query_words:
                # Stronger signal for name match
                if w in tool_name:
                    score += 5
                elif w in tool_desc:
                    score += 2
                elif w in params_text:
                    score += 1
                
            scored_tools.append((score, tool))
            
        # Sort by score descending
        scored_tools.sort(key=lambda x: -x[0])
        
        # Get top k
        top_tools = [tool for score, tool in scored_tools[:top_k]]
        return top_tools

