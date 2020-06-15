
import io
import json
import os
import pydoc
import re
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def metadata():

    def parse_docstring(docstring):
        headers = []
        lines = docstring.splitlines()
        indentation = min(filter(lambda s: s > 0, map(lambda s: len(s) - len(s.lstrip()), lines)))
        lines = list((s[indentation:] if len(s) > len(s.lstrip()) else s) for s in lines)
        docstring = '\n'.join(lines)
        tag_re = re.compile('(?<=\n)(Args|Arguments|Fields|Yields|Call arguments|Raises|Examples|Example|Usage|Input shape|Output shape|Returns|References):\n', re.MULTILINE)
        parts = tag_re.split(docstring)
        headers.append(('', parts.pop(0)))
        while len(parts) > 0:
            headers.append((parts.pop(0), parts.pop(0)))
        return headers

    def parse_arguments(arguments):
        result = []
        item_re = re.compile(r'^   ? ?(\*?\*?\w[\w.]*?\s*):\s', re.MULTILINE)
        content = item_re.split(arguments)
        if content.pop(0) != '':
            raise Exception('')
        while len(content) > 0:
            result.append((content.pop(0), content.pop(0)))
        return result

    def convert_code_block(description):
        lines = description.splitlines()
        output = []
        while len(lines) > 0:
            line = lines.pop(0)
            if line.startswith('>>>') and len(lines) > 0 and lines[0].startswith('>>>'):
                output.append('```')
                output.append(line)
                while len(lines) > 0 and lines[0] != '':
                    output.append(lines.pop(0))
                output.append('```')
            else:
                output.append(line)
        return '\n'.join(output)

    def update_argument(schema, name, description):
        attribute = None
        if not 'attributes' in schema:
            schema['attributes'] = []
        for current_attribute in schema['attributes']:
            if 'name' in current_attribute and current_attribute['name'] == name:
                attribute = current_attribute
                break
        if not attribute:
            attribute = {}
            attribute['name'] = name
            schema['attributes'].append(attribute)
        attribute['description'] = description

    json_path = os.path.join(os.path.dirname(__file__), '../src/keras-metadata.json')
    json_file = open(json_path)
    json_root = json.loads(json_file.read())
    json_file.close()

    for entry in json_root:
        name = entry['name']
        schema = entry['schema']
        if 'package' in schema:
            class_name = schema['package'] + '.' + name
            class_definition = pydoc.locate(class_name)
            if not class_definition:
                raise Exception('\'' + class_name + '\' not found.')
            docstring = class_definition.__doc__
            if not docstring:
                raise Exception('\'' + class_name + '\' missing __doc__.')
            headers = parse_docstring(docstring)
            for header in headers:
                key = header[0]
                value = header[1]
                if key == '':
                    description = convert_code_block(value)
                    description = description[:-1] if description.endswith('\n\n') else description
                    schema['description'] = description
                elif key == 'Args' or key == 'Arguments':
                    arguments = parse_arguments(value)
                    # for argument in arguments:
                    #     print('**' + argument[0] + '**')
                    #     print(argument[1])
                elif key == 'Call arguments':
                    pass
                elif key == 'Returns':
                    pass
                elif key == 'Input shape':
                    pass
                elif key == 'Output shape':
                    pass
                elif key == 'Example' or key == 'Examples' or key == 'Usage':
                    pass
                elif key == 'Raises':
                    pass
                elif key == 'References':
                    pass
                else:
                    print(key)

    json_file = open(json_path, 'w')
    json_data = json.dumps(json_root, sort_keys=True, indent=2)
    for line in json_data.splitlines():
        json_file.write(line.rstrip() + '\n')
    json_file.close()

def metadata_():

    def count_leading_spaces(s):
        ws = re.search(r'\S', s)
        if ws:
            return ws.start()
        else:
            return 0

    def process_list_block(docstring, starting_point, leading_spaces, marker):
        ending_point = docstring.find('\n\n', starting_point)
        block = docstring[starting_point:(None if ending_point == -1 else
                                          ending_point - 1)]
        # Place marker for later reinjection.
        docstring = docstring.replace(block, marker)
        lines = block.split('\n')
        # Remove the computed number of leading white spaces from each line.
        lines = [re.sub('^' + ' ' * leading_spaces, '', line) for line in lines]
        # Usually lines have at least 4 additional leading spaces.
        # These have to be removed, but first the list roots have to be detected.
        top_level_regex = r'^    ([^\s\\\(]+):(.*)'
        top_level_replacement = r'- __\1__:\2'
        lines = [re.sub(top_level_regex, top_level_replacement, line) for line in lines]
        # All the other lines get simply the 4 leading space (if present) removed
        lines = [re.sub(r'^    ', '', line) for line in lines]
        # Fix text lines after lists
        indent = 0
        text_block = False
        for i in range(len(lines)):
            line = lines[i]
            spaces = re.search(r'\S', line)
            if spaces:
                # If it is a list element
                if line[spaces.start()] == '-':
                    indent = spaces.start() + 1
                    if text_block:
                        text_block = False
                        lines[i] = '\n' + line
                elif spaces.start() < indent:
                    text_block = True
                    indent = spaces.start()
                    lines[i] = '\n' + line
            else:
                text_block = False
                indent = 0
        block = '\n'.join(lines)
        return docstring, block

    def process_docstring(docstring):
        # First, extract code blocks and process them.
        code_blocks = []
        if '```' in docstring:
            tmp = docstring[:]
            while '```' in tmp:
                tmp = tmp[tmp.find('```'):]
                index = tmp[3:].find('```') + 6
                snippet = tmp[:index]
                # Place marker in docstring for later reinjection.
                docstring = docstring.replace(
                    snippet, '$CODE_BLOCK_%d' % len(code_blocks))
                snippet_lines = snippet.split('\n')
                # Remove leading spaces.
                num_leading_spaces = snippet_lines[-1].find('`')
                snippet_lines = ([snippet_lines[0]] +
                                 [line[num_leading_spaces:]
                                 for line in snippet_lines[1:]])
                # Most code snippets have 3 or 4 more leading spaces
                # on inner lines, but not all. Remove them.
                inner_lines = snippet_lines[1:-1]
                leading_spaces = None
                for line in inner_lines:
                    if not line or line[0] == '\n':
                        continue
                    spaces = count_leading_spaces(line)
                    if leading_spaces is None:
                        leading_spaces = spaces
                    if spaces < leading_spaces:
                        leading_spaces = spaces
                if leading_spaces:
                    snippet_lines = ([snippet_lines[0]] +
                                     [line[leading_spaces:]
                                      for line in snippet_lines[1:-1]] +
                                     [snippet_lines[-1]])
                snippet = '\n'.join(snippet_lines)
                code_blocks.append(snippet)
                tmp = tmp[index:]

        # Format docstring lists.
        section_regex = r'\n( +)# (.*)\n'
        section_idx = re.search(section_regex, docstring)
        shift = 0
        sections = {}
        while section_idx and section_idx.group(2):
            anchor = section_idx.group(2)
            leading_spaces = len(section_idx.group(1))
            shift += section_idx.end()
            marker = '$' + anchor.replace(' ', '_') + '$'
            docstring, content = process_list_block(docstring,
                                                    shift,
                                                    leading_spaces,
                                                    marker)
            sections[marker] = content
            section_idx = re.search(section_regex, docstring[shift:])

        # Format docstring section titles.
        docstring = re.sub(r'\n(\s+)# (.*)\n',
                           r'\n\1__\2__\n\n',
                           docstring)

        # Strip all remaining leading spaces.
        lines = docstring.split('\n')
        docstring = '\n'.join([line.lstrip(' ') for line in lines])

        # Reinject list blocks.
        for marker, content in sections.items():
            docstring = docstring.replace(marker, content)

        # Reinject code blocks.
        for i, code_block in enumerate(code_blocks):
            docstring = docstring.replace(
                '$CODE_BLOCK_%d' % i, code_block)
        return docstring

    def split_docstring(docstring):
        headers = {}
        current_header = ''
        current_lines = []
        lines = docstring.split('\n')
        for line in lines:
            if line.startswith('__') and line.endswith('__'):
                headers[current_header] = current_lines
                current_lines = []
                current_header = line[2:-2]
                if current_header == 'Masking' or current_header.startswith('Note '):
                    headline = '**' + current_header + '**'
                    current_lines = headers['']
                    current_header = ''
                    current_lines.append(headline)
            else:
                current_lines.append(line)
        if len(current_lines) > 0:
            headers[current_header] = current_lines
        return headers

    def update_hyperlink(description):
        def replace_hyperlink(match):
            name = match.group(1)
            link = match.group(2)
            if link.endswith('.md'):
                if link.startswith('../'):
                    link = link.replace('../', 'https://keras.io/').rstrip('.md')
                else:
                    link = 'https://keras.io/layers/' + link.rstrip('.md')
                return '[' + name + '](' + link + ')'
            return match.group(0)
        return re.sub(r'\[(.*?)\]\((.*?)\)', replace_hyperlink, description)

    def update_argument(schema, name, lines):
        attribute = None
        if not 'attributes' in schema:
            schema['attributes'] = []
        for current_attribute in schema['attributes']:
            if 'name' in current_attribute and current_attribute['name'] == name:
                attribute = current_attribute
                break
        if not attribute:
            attribute = {}
            attribute['name'] = name
            schema['attributes'].append(attribute)
        description = '\n'.join(lines)
        description = update_hyperlink(description)
        attribute['description'] = description

    def update_arguments(schema, lines):
        argument_name = None
        argument_lines = []
        for line in lines:
            if line.startswith('- __'):
                line = line.lstrip('- ')
                colon = line.index(':')
                if colon > 0:
                    name = line[0:colon]
                    line = line[colon+1:].lstrip(' ')
                    if name.startswith('__') and name.endswith('__'):
                        if argument_name:
                            update_argument(schema, argument_name, argument_lines)
                        argument_name = name[2:-2]
                        argument_lines = []
            if argument_name:
                argument_lines.append(line)
        if argument_name:
            update_argument(schema, argument_name, argument_lines)
        return

    def update_examples(schema, lines):
        if 'examples' in schema:
            del schema['examples']
        summary_lines = []
        code_lines = None
        for line in lines:
            if line.startswith('```'):
                if code_lines != None:
                    example = {}
                    example['code'] = '\n'.join(code_lines)
                    if len(summary_lines) > 0:
                        example['summary'] = '\n'.join(summary_lines)
                    if not 'examples' in schema:
                        schema['examples'] = []
                    schema['examples'].append(example)
                    summary_lines = []
                    code_lines = None
                else:
                    code_lines = [ ]
            else:
                if code_lines != None:
                    code_lines.append(line)
                elif line != '':
                    summary_lines.append(line)

    def update_references(schema, lines):
        if 'references' in schema:
            del schema['references']
        references = []
        reference = ''
        for line in lines:
            if line.startswith('- '):
                if len(reference) > 0:
                    references.append(reference)
                reference = line.lstrip('- ')
            else:
                if line.startswith('  '):
                    line = line[2:]
                reference = reference + line
        if len(reference) > 0:
            references.append(reference)
        for reference in references:
            if not 'references' in schema:
                schema['references'] = []
            schema['references'].append({ 'description': reference })

    def update_input(schema, description):
        entry = None
        if 'inputs' in schema:
            for current_input in schema['inputs']:
                if current_input['name'] == 'input':
                    entry = current_input
                    break
        else:
            entry = {}
            entry['name'] = 'input'
            schema['inputs'] = []
            schema['inputs'].append(entry)
        if entry:
            entry['description'] = description

    def update_output(schema, description):
        entry = None
        if 'outputs' in schema:
            for current_output in schema['outputs']:
                if current_output['name'] == 'output':
                    entry = current_output
                    break
        else:
            entry = {}
            entry['name'] = 'output'
            schema['outputs'] = []
            schema['outputs'].append(entry)
        if entry:
            entry['description'] = description

    json_file = os.path.join(os.path.dirname(__file__), '../src/keras-metadata.json')
    json_data = open(json_file).read()
    json_root = json.loads(json_data)

    for entry in json_root:
        name = entry['name']
        schema = entry['schema']
        if 'package' in schema:
            class_name = schema['package'] + '.' + name
            class_definition = pydoc.locate(class_name)
            if not class_definition:
                raise Exception('\'' + class_name + '\' not found.')
            docstring = class_definition.__doc__
            if not docstring:
                raise Exception('\'' + class_name + '\' missing __doc__.')
            docstring = process_docstring(docstring)
            headers = split_docstring(docstring)
            if '' in headers:
                schema['description'] = '\n'.join(headers[''])
                del headers['']
            if 'Arguments' in headers:
                update_arguments(schema, headers['Arguments'])
                del headers['Arguments']
            if 'Input shape' in headers:
                update_input(schema, '\n'.join(headers['Input shape']))
                del headers['Input shape']
            if 'Output shape' in headers:
                update_output(schema, '\n'.join(headers['Output shape']))
                del headers['Output shape']
            if 'Examples' in headers:
                update_examples(schema, headers['Examples'])
                del headers['Examples']
            if 'Example' in headers:
                update_examples(schema, headers['Example'])
                del headers['Example']
            if 'References' in headers:
                update_references(schema, headers['References'])
                del headers['References']
            if 'Raises' in headers:
                del headers['Raises']
            if len(headers) > 0:
                raise Exception('\'' + class_name + '.__doc__\' contains unprocessed headers.')

    with io.open(json_file, 'w', newline='') as fout:
        json_data = json.dumps(json_root, sort_keys=True, indent=2)
        for line in json_data.splitlines():
            line = line.rstrip()
            fout.write(line)
            fout.write('\n')

def zoo():
    def download_model(type, file):
        file = os.path.expandvars(file)
        if not os.path.exists(file):
            folder = os.path.dirname(file)
            if not os.path.exists(folder):
                os.makedirs(folder)
            model = pydoc.locate(type)()
            model.save(file)
    if not os.environ.get('test'):
        os.environ['test'] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test'))
    download_model('tensorflow.keras.applications.DenseNet121', '${test}/data/keras/DenseNet121.h5')
    download_model('tensorflow.keras.applications.InceptionResNetV2', '${test}/data/keras/InceptionResNetV2.h5')
    download_model('tensorflow.keras.applications.InceptionV3', '${test}/data/keras/InceptionV3.h5')
    download_model('tensorflow.keras.applications.MobileNetV2', '${test}/data/keras/MobileNetV2.h5')
    download_model('tensorflow.keras.applications.NASNetMobile', '${test}/data/keras/NASNetMobile.h5')
    download_model('tensorflow.keras.applications.ResNet50', '${test}/data/keras/ResNet50.h5')
    download_model('tensorflow.keras.applications.VGG19', '${test}/data/keras/VGG19.h5')
    download_model('tensorflow.keras.applications.Xception', '${test}/data/keras/Xception.h5')

if __name__ == '__main__':
    command_table = { 'metadata': metadata, 'zoo': zoo }
    command = sys.argv[1]
    command_table[command]()
