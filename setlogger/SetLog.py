import ismrmrd
from collections import Counter
import re
import base64
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional



def parse_ice_header(text: str) -> Dict[str, Any]:
    text = text.replace('\x00', '')
    
    token_pattern = re.compile(
        r'<(?P<tag>[a-zA-Z0-9_]+)(?:\."(?P<key>[^"]*)")?>'  # <Tag> or <Tag."Key">
        r'|(?P<start>\{)'                                   # {
        r'|(?P<end>\})'                                     # }
        r'|"(?P<qval>(?:\\.|[^"\\])*)"'                     # "Quoted string"
        r'|(?P<uval>[^\s{}<">]+)'                           # Unquoted value
    )
    
    tokens = []
    for m in token_pattern.finditer(text):
        if m.group('tag'):
            tokens.append(('TAG', m.group('tag'), m.group('key')))
        elif m.group('start'):
            tokens.append(('START',))
        elif m.group('end'):
            tokens.append(('END',))
        elif m.group('qval') is not None:
            tokens.append(('VAL', m.group('qval')))
        elif m.group('uval') is not None:
            tokens.append(('VAL', m.group('uval')))

    pos = 0
    
    def next_tok():
        nonlocal pos
        if pos < len(tokens):
            t = tokens[pos]
            pos += 1
            return t
        return None

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def parse_block():
        elements = []
        while True:
            t = peek()
            if not t:
                break
            if t[0] == 'END':
                next_tok()  
                break
            elif t[0] == 'START':
                next_tok()
                elements.append(parse_block())
            elif t[0] == 'TAG':
                tag_t = next_tok()
                n = peek()
                val = None
                if n and n[0] == 'START':
                    next_tok()
                    val = parse_block()
                elif n and n[0] == 'VAL':
                    val = next_tok()[1]
                elements.append(('TAG_NODE', tag_t[1], tag_t[2], val))
            elif t[0] == 'VAL':
                elements.append(('VAL_NODE', next_tok()[1]))
            else:
                next_tok()
        return elements

    root_elements = []
    while peek():
        t = peek()
        if t[0] == 'START':
            next_tok()
            root_elements.append(parse_block())
        elif t[0] == 'TAG':
            tag_t = next_tok()
            n = peek()
            val = None
            if n and n[0] == 'START':
                next_tok()
                val = parse_block()
            elif n and n[0] == 'VAL':
                val = next_tok()[1]
            root_elements.append(('TAG_NODE', tag_t[1], tag_t[2], val))
        else:
            next_tok()

    def extract_single_value(nodes) -> Optional[str]:
        if not isinstance(nodes, list):
            return nodes
        for n in nodes:
            if n[0] == 'VAL_NODE':
                return n[1]
        return None

    def extract_array_items(nodes) -> List[Any]:
        items = []
        for n in nodes:
            if isinstance(n, list):
                v = extract_single_value(n)
                if v is not None:
                    items.append(v)
            elif n[0] == 'VAL_NODE':
                items.append(n[1])
        return items

    def ast_to_dict(ast_nodes) -> Dict[str, Any]:
        result = {}
        for node in ast_nodes:
            if not isinstance(node, tuple) or node[0] != 'TAG_NODE':
                continue
                
            tag, key, val = node[1], node[2], node[3]
            
            if tag == 'ParamLong':
                v = extract_single_value(val)
                if key: 
                    try: result[key] = int(v) 
                    except (ValueError, TypeError): result[key] = None
            elif tag == 'ParamDouble':
                v = extract_single_value(val)
                if key: 
                    try: result[key] = float(v) 
                    except (ValueError, TypeError): result[key] = None
            elif tag == 'ParamBool':
                v = extract_single_value(val)
                if key: 
                    result[key] = (str(v).lower() == 'true') if v is not None else None
            elif tag == 'ParamString':
                v = extract_single_value(val)
                if key: 
                    result[key] = str(v) if v is not None else None
            elif tag in ('ParamMap', 'XProtocol'):
                sub_dict = ast_to_dict(val) if isinstance(val, list) else {}
                if key:
                    result[key] = sub_dict
                else:
                    result.update(sub_dict)
            elif tag == 'ParamArray':
                arr = extract_array_items(val) if isinstance(val, list) else []
                if key:
                    result[key] = arr
        return result

    return ast_to_dict(root_elements)


class SetLog():
    """
    A set logger for ismrmrd.image and ismrmrd.acquisition classes
    acquisition logging is currently a WIP.
    """
    def __init__(self):
        self.headSet = {}
        self.metaSet ={}
        self.iceHeadSet = {}
        self.CustomSet = {}
        self.AcqHeadSet = {}
    def __str__(self):
        return self.get_SetLogs()

    def __repr__(self):
        return f"SetLog({self.headSet},{self.metaSet},{self.iceHeadSet},{self.CustomSet})"

    def get_SetLogs(self):
        s = ""
        if self.AcqHeadSet:
            s +="Acquistion Header Sets:" + "\n"+self.get_AcqHeadSet()+"\n"
        if self.headSet:
            s += "Image Head Sets:" + "\n"+self.get_headSet()+"\n"
        if self.metaSet:
            s +="Image Meta Sets:" + "\n"+self.get_metaSet()+"\n"
        if self.iceHeadSet:
            s +="Image IceHeader Sets:" + "\n"+self.get_iceHeadSet()+"\n"
        if self.CustomSet:
            s +="Custom Variable Sets:"  + "\n"+self.get_CustomSet()+"\n"
        return s

    def get_headSet(self):
        return self.__print_dicts(self.headSet)

    def get_metaSet(self):
        return self.__print_dicts(self.metaSet)

    def get_iceHeadSet(self):
            return self.__print_dicts(self.iceHeadSet)

    def get_AcqHeadSet(self):
        return self.__print_dicts(self.AcqHeadSet)

    def get_CustomSet(self):
        return self.__print_dicts(self.CustomSet)

    def __print_dicts(self, sets: dict):
        lines = [f"{key}: {self.__format_counter(val)}" for key, val in sets.items()]
        return "\n".join(lines) + "\n\n"

    def __format_counter(self,counter_obj):
        try:
            sorted_items = sorted(counter_obj.items())
        except TypeError:
            sorted_items = counter_obj.items()
        return ", ".join([f"{count}x({key})" for key, count in sorted_items])

    def __add_to_counter(self,s:dict,key,val):
        if isinstance(val,list):
            val = str(val)
        s.setdefault(key,Counter()).update([val])

    def __travserse_dict_and_log(self,data:dict,s:dict,var:str=None):
        for key in data.keys():
            if var!=None and str(key)==var:
                return data[key]
            if isinstance(data[key],dict):
                s.setdefault(key,{})
                self.__travserse_dict_and_log(data[key],s)
            else:
                self.__add_to_counter(s,key,data[key])
        return None

    def log_ImageHead(self,item,var:str=None):
        head_attrs = str(item._head).splitlines()
        for attr in head_attrs:
            attr = attr.split(":")
            self.headSet.setdefault(attr[0],Counter()).update([attr[1][1:]])

    def log_IceMiniHeader(self,iceHead):
        """
        This is also called in log_ImageMeta no need to double call.
        """
        iceHead_txt = base64.b64decode(iceHead).decode('utf-8')
        parsed = parse_ice_header(iceHead_txt)
        self.__travserse_dict_and_log(parsed,self.iceHeadSet)

    def log_ImageMeta(self,item):
        if isinstance(item,ismrmrd.meta.Meta):
            meta = item
        else:
            meta = ismrmrd.Meta.deserialize(item.attribute_string)
        for key in meta.keys():
            if key=="IceMiniHead":
                self.get_IceMiniHeader(meta[key])
            else:
                self.__add_to_counter(self.metaSet,key,meta[key])

    def log_ImageAll(self,item):
        if not isinstance(item,ismrmrd.image.Image):
            print("Error: instance is not a valid ismrmrd image")
            return
        self.log_ImageHead(item)
        self.log_ImageMeta(item)

    def __get_var_from_meta(self,meta,var_name:str):
        try:
            val = meta[var_name]
        except:
            val = ""
        return val

    def __get_var_from_head(self,head,var_name:str):
        head_attrs = str(head).splitlines()
        val = ""
        for attr in head_attrs:
            if var_name in attr:
                val = str(attr.split(":")[1])
        return val

    def __get_var_from_iceHead(self,iceHead,var_name:str):
        parsed = parse_ice_header(iceHead)
        self.__travserse_dict_and_log(parsed,self.CustomSet,var=var_name)
        return None

    def log_Var(self,item,var_name:str):
        if isinstance(item,ismrmrd.meta.Meta):
            self.CustomSet.setdefault(var_name,Counter()).update([self.__get_var_from_meta(item,var_name)])
        elif isinstance(item,ismrmrd.image.ImageHeader):
            self.CustomSet.setdefault(var_name,Counter()).update([self.__get_var_from_head(item,var_name)])
        elif isinstance(item,str):
            print("SetLog.log_Var: Item was type string, assuming iceMiniHeader")
            self.__get_var_from_iceHead(item,var_name)


    def log_AcquisitionHead(self, item):
        """
        WIP
        Log attributes from an Acquisition or AcquisitionHeader object
        """
        head = getattr(item, '_head', item)
        head_attrs = str(head).splitlines()
        for attr in head_attrs:
            parts = attr.split(":")
            if len(parts) >= 2:
                key = parts[-2].strip()
                val = parts[-1].strip()
                self.AcqHeadSet.setdefault(key, Counter()).update([val])

    def log_AcquisitionAll(self, item):
        """
        WIP
        Log header information for an ismrmrd Acquisition.
        """
        if not (isinstance(item, (ismrmrd.acquisition.Acquisition, ismrmrd.Acquisition)) or hasattr(item, '_head')):
            print("Error: instance is not a valid ismrmrd acquisition")
            return
        self.log_AcquisitionHead(item)

    def log_Acquisition(self, item):
        """alias for log_AcquisitionAll"""
        self.log_AcquisitionAll(item)
