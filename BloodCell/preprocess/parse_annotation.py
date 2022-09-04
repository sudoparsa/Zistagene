import xml.etree.ElementTree as ET


def parse_annotation(ann_dir, labels=None):
    all_imgs = []
    seen_labels = {}
    img = {'object': []}
    tree = ET.parse(ann_dir)

    for elem in tree.iter():
        if 'width' in elem.tag:
            img['width'] = int(elem.text)
        if 'height' in elem.tag:
            img['height'] = int(elem.text)
        if 'object' in elem.tag or 'part' in elem.tag:
            obj = {}

            for attr in list(elem):
                if 'name' in attr.tag:
                    obj['name'] = attr.text

                    if obj['name'] in seen_labels:
                        seen_labels[obj['name']] += 1
                    else:
                        seen_labels[obj['name']] = 1

                    if len(labels) > 0 and obj['name'] not in labels:
                        break
                    else:
                        img['object'] += [obj]

                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            obj['xmin'] = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            obj['ymin'] = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            obj['xmax'] = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            obj['ymax'] = int(round(float(dim.text)))

    if len(img['object']) > 0:
        all_imgs += [img]

    return all_imgs, seen_labels
