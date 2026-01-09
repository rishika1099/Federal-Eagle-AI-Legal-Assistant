# usc_parser.py
"""
Universal USC XML Parser
Parses ALL USC titles from usc folder into a single comprehensive JSON database
"""

import xml.etree.ElementTree as ET
import json
import re
from pathlib import Path
from typing import List, Dict
import argparse


def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = text.strip()
    return text


def extract_text_recursive(element):
    """Recursively extract all text from an element and its children"""
    texts = []
    
    if element.text:
        texts.append(element.text)
    
    for child in element:
        # Skip certain tags
        if child.tag not in ['{http://xml.house.gov/schemas/uslm/1.0}notes', 
                             '{http://xml.house.gov/schemas/uslm/1.0}note']:
            texts.append(extract_text_recursive(child))
        if child.tail:
            texts.append(child.tail)
    
    return ' '.join(filter(None, texts))


def parse_usc_file(xml_file: str) -> List[Dict]:
    """Parse a single USC title XML file"""
    
    print(f"Parsing: {Path(xml_file).name}")
    
    # Parse XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Define namespace
    ns = {'uslm': 'http://xml.house.gov/schemas/uslm/1.0',
          'dc': 'http://purl.org/dc/elements/1.1/'}
    
    # Get title number
    title_elem = root.find('.//dc:title', ns)
    title_match = re.search(r'Title (\d+[A-Za-z]*)', title_elem.text) if title_elem is not None else None
    title_num = title_match.group(1) if title_match else "Unknown"
    
    # Get title name
    title_heading = root.find('.//uslm:title/uslm:heading', ns)
    title_name = clean_text(title_heading.text) if title_heading is not None else "Unknown Title"
    
    print(f"  Title {title_num}: {title_name}")
    
    sections = []
    section_count = 0
    
    # Find all sections
    for section in root.findall('.//uslm:section', ns):
        try:
            # Get section number
            num_elem = section.find('uslm:num', ns)
            if num_elem is None:
                continue
            
            section_text = clean_text(num_elem.text)
            # Extract just the number (handle formats like "§ 1." or "[§ 1.")
            section_match = re.search(r'§?\s*\[?(\d+[A-Za-z]*)', section_text)
            if not section_match:
                continue
            
            section_num = section_match.group(1)
            
            # Skip if repealed/omitted
            heading_elem = section.find('uslm:heading', ns)
            if heading_elem is not None:
                heading_text = clean_text(heading_elem.text)
                if 'Repealed' in heading_text or 'Omitted' in heading_text:
                    continue
                section_title = heading_text
            else:
                section_title = "No title"
            
            # Extract section content
            content = extract_text_recursive(section)
            content = clean_text(content)
            
            # Limit content length
            if len(content) > 10000:
                content = content[:10000] + "..."
            
            # Get chapter info (traverse up to find chapter)
            chapter_num = 1
            chapter_title = "General Provisions"
            
            parent = section
            for _ in range(10):  # Look up 10 levels max
                parent_elem = parent.find('..')
                if parent_elem is None:
                    break
                
                if 'chapter' in parent_elem.tag:
                    ch_num_elem = parent_elem.find('uslm:num', ns)
                    ch_heading_elem = parent_elem.find('uslm:heading', ns)
                    
                    if ch_num_elem is not None:
                        ch_num_text = clean_text(ch_num_elem.text)
                        ch_match = re.search(r'(\d+)', ch_num_text)
                        if ch_match:
                            chapter_num = int(ch_match.group(1))
                    
                    if ch_heading_elem is not None:
                        chapter_title = clean_text(ch_heading_elem.text)
                    
                    break
                
                parent = parent_elem
            
            sections.append({
                "title": title_num,
                "title_name": title_name,
                "chapter": chapter_num,
                "chapter_title": chapter_title,
                "Section": section_num,
                "section_title": section_title,
                "section_desc": content,
                "citation": f"{title_num} U.S.C. § {section_num}"
            })
            
            section_count += 1
            
        except Exception as e:
            print(f"    Error parsing section: {e}")
            continue
    
    print(f"  Extracted {section_count} sections\n")
    return sections


def parse_all_usc_files(input_dir: str = 'usc', output_file: str = 'usc_complete.json'):
    """Parse all USC XML files in the usc directory"""
    
    print("\nUSC UNIVERSAL PARSER")
    print("Parsing all U.S. Code titles from usc folder\n")
    
    input_path = Path(input_dir)
    
    # Check if directory exists
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' not found")
        print("Please create a 'usc' folder and place all USC XML files there")
        return
    
    # Find all USC XML files
    xml_files = sorted(input_path.glob('usc*.xml'))
    
    if not xml_files:
        print(f"No USC XML files found in {input_dir}")
        print("Please place USC XML files (usc01.xml, usc18.xml, etc.) in the usc folder")
        return
    
    print(f"Found {len(xml_files)} USC title files\n")
    
    all_sections = []
    
    # Parse each file
    for xml_file in xml_files:
        try:
            sections = parse_usc_file(str(xml_file))
            all_sections.extend(sections)
        except Exception as e:
            print(f"  ERROR processing {xml_file.name}: {e}\n")
            continue
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_sections, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\nPARSING COMPLETE")
    print(f"\nTotal sections extracted: {len(all_sections)}")
    print(f"Output file: {output_file}")
    
    # Show breakdown by title
    print("\nBreakdown by Title:")
    title_counts = {}
    for section in all_sections:
        title = section['title']
        title_counts[title] = title_counts.get(title, 0) + 1
    
    for title in sorted(title_counts.keys(), key=lambda x: (len(x), x)):
        count = title_counts[title]
        # Get title name from first section
        title_name = next(s['title_name'] for s in all_sections if s['title'] == title)
        print(f"  Title {title:>3}: {count:>5} sections - {title_name}")
    
    print(f"\nTotal: {len(all_sections)} sections across {len(title_counts)} titles\n")


def main():
    parser = argparse.ArgumentParser(
        description='Parse all USC XML files from usc folder into comprehensive JSON database'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='usc',
        help='Directory containing USC XML files (default: usc)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='usc_complete.json',
        help='Output JSON filename (default: usc_complete.json)'
    )
    
    args = parser.parse_args()
    
    parse_all_usc_files(args.input_dir, args.output)


if __name__ == "__main__":
    main()