import xml.etree.ElementTree as ET

import xml.etree.ElementTree as ET

def overlay_svg_difference(base_svg_path, diff_svg_path, output_svg_path):
    # Load the base and the difference SVGs
    base_tree = ET.parse(base_svg_path)
    diff_tree = ET.parse(diff_svg_path)
    
    base_root = base_tree.getroot()
    diff_root = diff_tree.getroot()

    # SVG namespace
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Iterate through elements in the diff SVG
    for diff_elem in diff_root.findall('.//svg:*', ns):
        diff_id = diff_elem.attrib.get('id')
        if diff_id:
            # Find the corresponding element in the base SVG by ID
            base_elem = base_root.find(f".//*[@id='{diff_id}']", ns)

            # If the element is different (or doesn't exist), add it to the base SVG
            if base_elem is None or not elements_are_equal(base_elem, diff_elem):
                # If base_elem exists, remove it
                if base_elem is not None:
                    base_root.remove(base_elem)
                # Add the element from the diff SVG
                base_root.append(diff_elem)

    # Write the result to the output file
    base_tree.write(output_svg_path)

def elements_are_equal(elem1, elem2):
    """
    Compares two XML elements for equality based on their attributes and text.
    This function assumes that elements have unique IDs.
    """
    # Compare tag, attributes, and text content
    if elem1.tag != elem2.tag:
        return False
    if elem1.attrib != elem2.attrib:
        return False
    if (elem1.text or '').strip() != (elem2.text or '').strip():
        return False
    # Optionally, you could compare children if they have any, but often agent paths don't
    return True

# Usage example
base_svg = '/home/roman/Documents/CommonRoad/PA_CommonRoad/planner/Frenet/figs/KL0Mergefig_7.svg'
overlay_svgs = '/home/roman/Documents/CommonRoad/PA_CommonRoad/planner/Frenet/figs/KL0Mergefig_50.svg'#, 
#                '/home/roman/Documents/CommonRoad/PA_CommonRoad/planner/Frenet/figs/KL0Mergefig_50.svg',
#                '/home/roman/Documents/CommonRoad/PA_CommonRoad/planner/Frenet/figs/KL0Mergefig_70.svg']
output_svg = 'merged_output.svg'

overlay_svg_difference(base_svg, overlay_svgs, output_svg)
