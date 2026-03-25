import re
puncts = ['?', '!', ':', ';']
replace_token = {'~': '\\sim', '∼': '\\sim', 'π': '\\pi', '\\varpi': '\\pi', 'α':'\\alpha', 'λ': '\\lambda', '°': ['^', '\\circ'], 'θ': '\\theta', '\\vartheta': '\\theta', '¬': '\\neg',
                 '∈': '\\in', '≤': '\\le', '\\leq': '\\le', '\\leqq': '\\le', '→': '\\rightarrow', '\\to': '\\rightarrow', '∞': '\\infty', 
                 '∀': '\\forall', 'δ': '\\delta', '≥': '\\ge', '\\geq': '\\ge', '\\geqq': '\\ge', '\\geqslant': '\\ge', '∧': '\\wedge', '\\bigwedge': '\\wedge', '∨': '\\vee', '\\bigvee': '\\vee',
                 '∂': '\\partial', 'φ': '\\phi', 'ϕ': '\\phi', '\\varphi': '\\phi', 'β': '\\beta', 'σ': '\\sigma', '\\varsigma': '\\sigma', '\\Sigma': '\\sigma', 
                 '∩': '\\cap', '\\bigcap': '\\cap', 'ε': '\\epsilon', 'ω': '\\omega', '∃': '\\exists', 'μ': '\\mu', 'γ': '\\gamma', '≡': '\\equiv', '∪': '\\cup',
                 '\\bigcup': '\\cup', '√': '\\sqrt', '`': '\'', 'τ': '\\tau', '±': '\\pm', '≠': '\\neq', '∫': '\\int', 
                 'ϵ': '\\varepsilon', '\\longleftarrow': '\\leftarrow', '⇒': '\\rightarrow', '\\longrightarrow': '\\rightarrow', 'ρ': '\\rho', '\\varrho': '\\rho', 
                 '⊆': '\\subseteq', '⊇': '\\supseteq', '⊂': '\\subset', '⊃': '\\supset', '⊈': '\\nsubseteq', '⊉': '\\nsupseteq', '∉': '\\notin', 
                 '•': '\\bullet', '∇': '\\nabla', '´': '\\acute', '∑': '\\sum', 'ℝ': ['\\mathbb', '{', 'r', '}'], '⟩': '\\rangle', 
                 '⟨': '\\langle', '\\mid': '|', '\\vert': '|', 'ξ': '\\xi', '∘': '\\circ', 'η': '\\eta', '\\vareta': '\\eta', '↑': '\\uparrow', 
                 '↓': '\\downarrow', '←': '\\leftarrow', '‖': '\|', '∥': '\|', '⊕': '\\oplus', '\\bigoplus': '\\oplus', '≈': '\\approx', 'ζ': '\\zeta',
                 'ℤ': ['\\mathbb', '{', 'z', '}'], 'º': ['^', '\\circ'], '∠': '\\angle', '∅': '\\emptyset', 'ø': '\\emptyset', '◦': '\\circ', 
                 '⋯': '\\ldots', '⊗': '\\otimes', '\\bigotimes': '\\otimes', '‘': '\'', '’': '\'', '⇔': '\\leftrightarrow', '⊥': '\\perp', 'µ': '\\mu', 'χ': '\\chi', 
                 '─': '-', 'ℕ': ['\\mathbb', '{', 'n', '}'], '“': '"', '”': '"', '↦': '\\mapsto', '↔': '\\leftrightarrow', 
                 '⟹': '\\rightarrow', '∆': '\\delta', 'κ': '\\kappa', '\\varkappa': '\\kappa', '⊢': '\\vdash', '\cosθ': ['\\cos', '(', '\\theta', ')'], '\sinθ': ['\\sin', '(', '\\theta', ')'], 
                 '≅': '\\cong', '⌊': '\\lfoor', '⌋': '\\rfloor', '△': '\\triangle', '\\bigtriangleup': '\\triangle', '\\bigtriangledown': '\\triangledown', 
                 '，': ',', 'ℚ': ['\\mathbb', '{', 'q', '}'], '≦': '\\le', '\\leqslant': '\\le', '│': '|', '□': '\\square', '）': ')', '＝': '=', '⌈': '\\lceil', '⌉': '\\rceil', '\\bar': '\\overline', 
                 '\\ne': '\\neq', '\\lvert': '|', '\\rvert': '|', '\\land': '\\wedge', '\\gt': '>','\\lt': '<', '\\cr': '\\\\', '\\prime': '\'', '\\colon': ':', '\\bmod': '\\mod', 
                 '\\lbrace': '\{', '\\rbrace': '\}', '\\widehat': '\\hat', '\\widetilde': '\\tilde', '\\backslash': '\\', '\\longleftrightarrow': '\\leftrightarrow', '\\varnothing': '\\emptyset', '\\dbinom': '\\binom', '\\tbinom': '\\binom',
                 '\\bot': '\\perp', '\\lbrack': '[', '\\rbrack': ']', '\\parallel': '\|', '\\longmapsto': '\\mapsto', '\\intercal': 't', '\\bigsqcup': '\\sqcup', '\\bigtriangleup': '\\triangle', '\\intop': '\\int', '\\eps': '\\epsilon', 
                 '\\gets': '\\leftarrow', '\\bbbk': ['\\mathbb', '{', 'k', '}'], '\\thicksim': '\\sim', '\\biguplus': '\\uplus', '\\vartriangle': '\\triangle', '\\triangleleft': '\\lhd', '\\triangleright': '\\rhd', '\\vartriangleleft': '\\lhd',
                 '\\arrowvert': '|', 'é': ['\\\'', '{', 'e', '}'], '\\acute': '\\\'', '\\varsubsetneq': '\\subsetneq', '∞': '\infty', '–':'-', '−': '-', '×': '*'}

def split(text):
    # splits text at period and puncts symbols
    text_spitted = text.split('. ')  # keep expressions like e.g. or q.e.d.
    splitted = []
    for t in text_spitted:
        current = [t]
        search_puncts = puncts[:]
        while len(search_puncts) > 0:
            punct = search_puncts.pop()
            current_splitted = []
            for c in current:
                if punct in c:
                    current_splitted.extend(c.split(punct))
                else:
                    current_splitted.append(c)
            current = current_splitted
        splitted.extend(current)
    splitted = [s.strip() for s in splitted]
    splitted = [s for s in splitted if s != '']
    return [s + '\n' for s in splitted]


# https://stackoverflow.com/a/49186435
def divide(lst, n):  # divide list in n parts of equal length (more or less)
    p = len(lst) // n
    if len(lst) - p > 0:
        return [lst[:p]] + divide(lst[p:], n - 1)
    else:
        return [lst]
    
def is_number(s):    
    try:    
        float(s)        
        return True    
    except ValueError:        
        pass  
    try:        
        import unicodedata   
        unicodedata.numeric(s)   
        return True    
    except (TypeError, ValueError):        
        pass    
        return False

voc_list = []
data_path = './vocabulary'
with open(f'{data_path}/voc_list_new.txt', 'r', encoding='utf-8') as file:
    for line in file:
        voc_list.append(line.split('\t')[0])

class LatexTokenizer:
    """Tokenizer for splitting LaTeX expressions into tokens"""

    def tokenize(self, eq):
        result = []
        last_token = ''
        for char in list(eq):
            if char == '\\' and last_token != '\\':
                if last_token:
                    result.extend(self.operator_check(last_token))
                last_token = '\\'
            elif last_token and last_token[0] in ['\\']:
                if last_token == '\\' and char in ['{', '}', '|', '#', '$', '%', '&', '_', ',', '!', '\\']:
                    last_token += char
                    continue
                if last_token in ['\\{', '\\}', '\\|', '\\#', '\\$', '\\%', '\\&', '\\_', '\\,', '\\\'', '\\!', '\\\\']: #if \\ is followed by special characters reserved by latex
                    result.extend(self.operator_check(last_token))
                    last_token = char
                    continue

                if char.isalpha(): #if "\\" is followed by alphabet
                    last_token += char
                else: #if "\\" is followed by other things
                    if last_token:
                        result.extend(self.operator_check(last_token))
                    last_token = char
            elif char.isdigit() or char == '.':  # TODO support , as decimal seperator?
                if last_token.replace('.', '').isdigit(): #if current last_token is all digits
                    last_token += char
                elif char.isdigit() and last_token == '.': #avoid splitting .1234
                    last_token += char
                else:
                    if last_token:
                        result.extend(self.operator_check(last_token))
                    last_token = char
            else:
                if last_token:
                    result.extend(self.operator_check(last_token))
                last_token = char
        result.extend(self.operator_check(last_token))
        return result
    
    def operator_check(self, token): #check if token is a official latex op that appears in voc_list
        token_list = []
        if token.lower() in voc_list: #if token exists in voc list
            token_list.append(token.lower())
            return token_list
        else:
            if token.startswith('\\'): #if the token startswith \\ but not in voc list, means it might be a local operator or rare operator, split it into digits
                for i in range(1,len(token)):
                    token_list.append(token[i].lower())
                return token_list
            else:
                if token == ' ' or is_number(token):
                    if len(token) <= 7: #for numbers, only keep first 7 digits
                        token_list.append(token.lower())
                    else:
                        if token[6] != '.': 
                            token_list.append(token[:7].lower()) 
                        else: #if the last digit is a ".", keep one digit after the "."
                            token_list.append(token[:8].lower())
                else:
                    token_list.append('spec') #not startswith \\, also not in voc list, might be a rare op
                return token_list
            
    def remove_whitespace_tokens(self, eq):
        tokenized = self.tokenize(eq)
        cleaned = [t for t in tokenized if t.strip() not in ['']]
        return cleaned
    
    def merge_tokens(self, eq): #merge \\begin{xxxx}{}
        eq = self.remove_whitespace_tokens(eq)
        result = []
        sign_list = ['|', '\\|']
        matrix_sign = ['vmatrix', 'matrix', 'bmatrix', 'pmatrix', 'Vmatrix', 'Bmatrix', 'array',
                       'align', 'align*', 'split', 'cases', 'eqnarray', 'eqnarray*',
                       'equation', 'equation*', 'alignat', 'alignat*', 'aligned',
                       'alignedat', 'gather', 'gather*', 'gathered', 'multline',
                       'multline*', 'smallmatrix', 'subarray']
        i = 0
        while i < len(eq):
            if eq[i] in ['\\begin', '\\end', '\\']:
                found = False
                for op in matrix_sign:
                    if i+len(op)<len(eq) and ''.join(eq[i:i+len(op)+3]) == f'{eq[i]}{{{op}}}':
                        
                        if op in ['vmatrix', 'matrix', 'bmatrix', 'pmatrix', 'Vmatrix', 'Bmatrix', 'smallmatrix']: #matrix class
                            result.append(f'{eq[i]}{{matrix}}')
                            i += len(op)+3
                            found = True
                            break
                        elif op in ['align', 'align*', 'alignat', 'alignat*', 'aligned', 'alignedat', 
                                    'eqarray', 'eqarray*', 'equation', 'equation*', 'gather', 'gather*', 
                                    'gathered','multline', 'multline*', 'split', ]:
                            i += len(op)+3
                            found = True
                            break
                        elif op in ['array', 'subarray']:
                            result.append(f'{eq[i]}{{array}}')
                            
                            i += len(op)+3
                            found = True
                            break
                        else:
                            result.append(f'{eq[i]}{{{op}}}')
                            i += len(op)+3
                            found = True
                            break
                if found == False:
                    result.append(eq[i])
                    i += 1
            else:
                result.append(eq[i])
                i += 1
        return result
    
    def remove_textop(self, eq): #remove \text{abc}, but if the formula is written as \text abc, this function will not remove \text
        cleaned = self.merge_tokens(eq)
        text_op_list = ['\\text', '\\textbf', '\\textit', '\\textrm', '\\textsf', '\\textstyle', '\\texttt', '\\mbox', '\\mathcal', '\\mathbf', '\\mathfrak', '\\mathhscr', '\\mathsf', '\\boldsymbol', 
                        '\\mathtt', '\\pmb', '\\frak', '\\hbox', '\\mathit', '\\displaystyle', '\\rm', '\\bf', '\\large', '\\small', '\\cal', '\\tt', '\\normalsize', '\\textstyle', '\\fbox', '\\mathbin',
                        '\\scriptstyle', '\\rlap', '\\scriptsize', '\\smash', '\\mathord', '\\srrc'] #'mathbb', '\\mathrm',
        text_indices = []
        for op in text_op_list:
            text_indices.extend(find_text_indices(cleaned, op))
        text_indices.sort()
        i = 0
        result = []
        while i < len(cleaned):
            if i in text_indices:
                i += 1
            else:
                result.append(cleaned[i])
                i += 1
        return result
    
    def remove_color(self, eq): #remove \\color{red}{xxxxxx}
        eq = self.remove_textop(eq)
        result = []
        indices = []
        found_color = 0
        i = 0
        while i < len(eq):
            if eq[i] in ['\\color']:
                found_color = 1
                stack = 0
                num_rightbrace = 0
                token_idx = i
            elif eq[i] == '{' and found_color == 1:
                stack += 1
            elif eq[i] == '}' and found_color == 1:
                stack -= 1
                if stack == 0:
                    num_rightbrace += 1
                if num_rightbrace == 1:
                    if eq[i+1] != '{':
                        indices.extend([i for i in range(token_idx, i+1)])
                        found_color = 0
                    else:
                        indices.extend([i for i in range(token_idx, i+1)])
                        indices.append(i+1)
                elif num_rightbrace == 2:
                    found_color = 0
                    indices.append(i)
            i+=1
        for i in range(len(eq)):
            if i in indices:
                i+=1
            else:
                result.append(eq[i])
        return result
    
    def delete_all(self, eq): #delete \\label{xxxx}
        i = 0
        found = 0
        indices = []
        result = []
        while i<len(eq):
            if eq[i] in ['\\label', '\\ref', '\\eqref']:
                token_idx = i
                found = 1
                stack = 0
            elif eq[i] == '{' and found == 1:
                stack += 1
            elif eq[i] == '}' and found == 1:
                stack -= 1
                if stack == 0:
                    indices.extend([i for i in range(token_idx, i+1)])
                    found = 0
            i+=1
        for i in range(len(eq)):
            if i in indices:
                i+=1
            else:
                result.append(eq[i])
        return result

    def number_standarization(self, num): #if number like 12.34.56 or .123 appears, convet it to similar decimal
        result = []
        if num[0] == '.':
            num = '0' + num
        dot_count = 0
        for char in num:
            if char == '.':
                dot_count += 1
                if dot_count >= 2:
                    return ''.join(result)
            result.append(char)
        return ''.join(result)

    def fraction_check(self, i, eq): #process nonstandard fraction
        result = []
        if eq[i+1] == '{':
            add_i = 1
            result.append(eq[i])
            return result, add_i
        else:
            if is_number(eq[i+1]) and len(eq[i+1])>=2:
                result.append('\\frac')
                result.extend(['{', f'{eq[i+1][0]}', '}', '{', f'{eq[i+1][1]}', '}'])
                if len(eq[i+1])>2:
                    result.append(f'{eq[i+1][2:]}')
                add_i = 2
            elif is_number(eq[i+1]) and len(eq[i+1])==1:
                if eq[i+2] in voc_list and eq[i+2] != '{' and not is_number(eq[i+2]):
                    result.append('\\frac')
                    result.extend(['{', f'{eq[i+1]}', '}', '{', f'{eq[i+2]}', '}'])
                    add_i = 3
                elif is_number(eq[i+2]):
                    if len(eq[i+2]) == 1: #\frac 1 2
                        result.append('\\frac')
                        result.extend(['{', f'{eq[i+1]}', '}', '{', f'{eq[i+2]}', '}'])
                        add_i = 3
                    elif len(eq[i+2])>1: #\frac 1 123
                        result.append('\\frac')
                        result.extend(['{', f'{eq[i+1]}', '}', '{', f'{eq[i+2][0]}', '}'])
                        result.append(f'{eq[i+2][1:]}')
                        add_i = 3
                elif eq[i+2] == '{': #\frac1{abc}
                    result.append('\\frac')
                    result.extend(['{', f'{eq[i+1]}', '}'])
                    add_i = 2
            elif eq[i+1] in voc_list and eq[i+1] != '{' and not is_number(eq[i+1]):
                if eq[i+2] in voc_list and eq[i+2] != '{' and not is_number(eq[i+2]): #\fracxy
                    result.append('\\frac')
                    result.extend(['{', f'{eq[i+1]}', '}', '{', f'{eq[i+2]}', '}'])
                    add_i = 3
                elif is_number(eq[i+2]): #\fracx1
                    if len(eq[i+2]) == 1:
                        result.append('\\frac')
                        result.extend(['{', f'{eq[i+1]}', '}', '{', f'{eq[i+2]}', '}'])
                        add_i = 3
                    elif len(eq[i+2])>1: #\fracx123
                        result.append('\\frac')
                        result.extend(['{', f'{eq[i+1]}', '}', '{', f'{eq[i+2][0]}', '}'])
                        result.append(f'{eq[i+2][1:]}')
                        add_i = 3
                elif eq[i+2] == '{': #\fracx{abc}
                    result.append('\\frac')
                    result.extend(['{', f'{eq[i+1]}', '}'])
                    add_i = 2
        return result, add_i
    
    def op_add_brace(self, i, eq): #add brace for situation like x_1, e^3
        result = []
        if eq[i+1] == '{':
            result.append(eq[i])
            add_i = 1
            return result, add_i
        else:
            if is_number(eq[i+1]):
                result.append(eq[i])
                result.extend(['{', f'{eq[i+1][0]}', '}'])
                if len(eq[i+1])>1:
                    result.append(f'{eq[i+1][1:]}')
                add_i = 2
            elif eq[i+1] in voc_list and eq[i+1] != '{' and not is_number(eq[i+1]):
                result.append(eq[i])
                result.extend(['{', f'{eq[i+1]}', '}'])
                add_i = 2
        return result, add_i
    
    def num_transform(self, num): #convert numbers to exponential
        num_dict = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        out = []
        if '.' not in num:
            for char in num:
                if char in num_dict:
                    out.append(char)
                else:
                    out.append('spec')
            out.extend(['e', '{', '0', '}'])
            return out
        elif '.' in num:
            num_list = num.split('.')
            if len(num_list) == 2:
                num = num_list[0]+num_list[1]
                for char in num:
                    if str(char) in num_dict:
                        out.append(char)
                    else:
                        out.append('spec')
                out.extend(['e', '{', '-', f'{len(num_list[1])}', '}'])
                return out
            else:
                for char in num:
                    if char in num_dict:
                        out.append(char)
                    else:
                        out.append('spec')
                return out
        
    def standarization(self, eq):
        if '\\color' in eq: #remove color
            try:
                eq = self.remove_color(eq)
            except:
                eq = self.remove_textop(eq)
        else:
            eq = self.remove_textop(eq)
        if '\\label' in eq or '\\ref' in eq or '\\eqref' in eq: #remove \\label{xxxxxx}
            try:
                eq = self.delete_all(eq)
            except:
                pass
        eq1 = []
        result = []
        i = 0
        while i < len(eq):
            if eq[i] in ['\\frac', '\\dfrac', '\\tfrac', '\\cfrac', '\\genfrac']: #unify fraction operators
                try:
                    out, add_i = self.fraction_check(i, eq)
                    eq1.extend(out)
                    i += add_i
                except:
                    eq1.append(eq[i].lower())
                    i += 1
                    continue
            elif eq[i] in ['_', '^']: #add brace for x_1, e^3
                try:
                    out, add_i = self.op_add_brace(i, eq)
                    eq1.extend(out)
                    i += add_i
                except:
                    eq1.append(eq[i].lower())
                    i += 1
                    continue
            else:
                eq1.append(eq[i].lower()) #all use lower case
                i += 1
        i = 0
        while i < len(eq1):
            if eq1[i] in ['*', '·', '⋅', '\\cdot', '・', '\\cdotp', '\\centerdot', '\\times', '⊗', '\\bigotimes', '\\otimes', '\\ast', '∗']:
                result.append('*')
                i += 1
            elif eq1[i] in ['/', '\\div', '÷']:
                result.append('/')
                i += 1
            elif eq1[i] in ['\\quad', '\\qquad', '\\newline', '\\hskip']:
                result.extend(['\\\\'])
                i += 1
            elif eq1[i] in ['&']:
                if i+1 < len(eq1) and eq1[i+1] == '=':
                    i += 1
                else:
                    result.append(eq1[i])
                i += 1
            elif eq1[i] in ['\\', '\\backslash', '\\left', '⌊', '\\acute', '´', '\\right', '$', '\\,', '\\;', '\\!', '\\:', '\\bigg', '\\bigl', '\\bigr', '\\hline', '\\biggl', '\\biggr','\\hfill', '\\space', '\\thinspace', '\\enspace', '\\box', '\\it', '\\sf',
                           '\\hbar', '\\middle', '\\nolimits', '\\heartsuit', '\\spadesuit', '\\clubsuit', '\\diamondsuit', '\\bigm', '\\tiny', '\\strut', '\\mathstrut', '\\bigstar', '\\biggm', '\u2061', '\u200B', '\u200E', '\t', '\u200c']:
                i += 1
            elif eq1[i] in ['\\text', '\\textbf', '\\textit', '\\textrm', '\\textsf', '\\textstyle', '\\texttt', '\\mbox', '\\mathcal', '\\mathbf', '\\mathfrak', '\\mathhscr', '\\mathsf', '\\boldsymbol', 
                           '\\mathtt', '\\pmb', '\\frak', '\\hbox', '\\mathit', '\\displaystyle', '\\rm', '\\bf', '\\large', '\\small', '\\cal', '\\tt', '\\normalsize', '\\textstyle', '\\fbox', '\\mathbin',
                        '\\scriptstyle', '\\rlap', '\\scriptsize', '\\smash', '\\mathord', '\\srrc']:
                i += 1
            elif eq1[i] in ['\\cdots', '\\dots', '\\dotsc', '\\dotsb', '\\dotsm', '\\dotso', '…']:
                result.append('\\ldots')
                i += 1
            elif eq1[i].replace('.', '').isdigit(): #process numbers
                result.append(self.number_standarization(eq1[i]))
                i += 1
            elif eq1[i] in replace_token:
                if type(replace_token[eq1[i]]) == list:
                    result.extend(replace_token[eq1[i]])
                else:
                    result.append(replace_token[eq1[i]])
                i += 1
            else:
                result.append(eq1[i].lower()) #all use lower case
                i += 1
        i = 0
        out = []
        
        while i < len(result):
            if is_number(result[i]):
                out.extend(self.num_transform(result[i])) 
            else:
                out.append(result[i])
            i += 1
        return out
    
    def count_textual_content(self, eq):
        text_len = 0
        for i in range(len(eq)):
            if i >= 1 and eq[i-1].isalpha() and eq[i].isalpha():
                text_len += 1
            elif eq[i] == 'spec':
                text_len += 1
        return text_len

    def plug_multi_in(self, eq): #optional, add multiplication sign in situation like xy -> x*y
        cleaned = self.standarization(eq)
        op_list = ['\\frac', '\\cfrac', '\\sqrt', '\\exp', '\\log', '\\ln', '\\lg',
                   '\\sin', '\\cos', '\\tan', '\\cot', '\\sec', '\\csc',
                    '\\arcsin', '\\arccos', '\\arctan', '\\arccot', '\\arcsec', '\\arccsc',
                    '\\sinh', '\\cosh', '\\tanh', '\\coth', '\\sech', '\\csch',
                    '\\arcsinh', '\\arccosh', '\\arctanh', '\\arccoth', '\\arcsech', '\\arccsch', 
                    '\\begin{vmatrix}', '\\begin{bmatrix}', '\\begin{matrix}', '\\begin{Vmatrix}', '\\begin{pmatrix}',
                    '\\overline', '\\underline', '\\widehat', '\\widetilde', '\\overrightarrow', '\\overleftarrow', '\\overbrace', '\\underbrace', 
                    '\\alpha', '\\beta', '\\delta', '\\epsilon', '\\eta', '\\gamma', '\\kappa', '\\lambda', '\\mu', '\\omega', '\\phi', 
                    '\\pi', '\\psi', '\\rho', '\\sigma', '\\tau', '\\theta', '\\xi', '\\zeta', '\\chi', '\\upsilon', '\\iota', '\\varepsilon', '\\varphi', '\\Delta', 
                    '\\Gamma', '\\Lambda', '\\Omega', '\\Phi', '\\Pi', '\\Psi', '\\Sigma', '\\Theta', 
                    '\\langle', '\\acute', '\\breve', '\\ddot', '\\grave', '\\tilde', '\\bar', '\\check', '\\dot', '\\hat', '\\vec']
        left_single = ['\\sqrt', '\\overline', '\\underline', '\\widehat', '\\widetilde', '\\overrightarrow', '\\overleftarrow', '\\overbrace', '\\underbrace',
                       '\\acute', '\\breve', '\\ddot', '\\grave', '\\tilde', '\\bar', '\\check', '\\dot', '\\hat', '\\vec'] #'^', '_'
        left_double = ['\\frac', '\\cfrac']
        left_token = [')', '\\end{vmatrix}', '\\end{bmatrix}', '\\end{matrix}', '\\end{Vmatrix}', '\\end{pmatrix}', 
                    '\\alpha', '\\beta', '\\delta', '\\epsilon', '\\eta', '\\gamma', '\\kappa', '\\lambda', '\\mu', '\\omega', '\\phi', 
                    '\\pi', '\\psi', '\\rho', '\\sigma', '\\tau', '\\theta', '\\xi', '\\zeta', '\\varepsilon', '\\varphi', '\\Delta', 
                    '\\Gamma', '\\Lambda', '\\Omega', '\\Phi', '\\Pi', '\\Psi', '\\Sigma', '\\Theta', '\\rangle', ]
        greek_letters = ['\\alpha', '\\beta', '\\delta', '\\epsilon', '\\eta', '\\gamma', '\\kappa', '\\lambda', '\\mu', '\\omega', '\\phi', 
                    '\\pi', '\\psi', '\\rho', '\\sigma', '\\tau', '\\theta', '\\xi', '\\zeta', '\\chi', '\\upsilon', '\\iota', '\\varepsilon', '\\varphi', '\\Delta', 
                    '\\Gamma', '\\Lambda', '\\Omega', '\\Phi', '\\Pi', '\\Psi', '\\Sigma', '\\Theta']
        
        single_indices = []
        for op in left_single:
            single_indices.extend(find_end_indices_one_aug(cleaned, op))
        double_indices = []
        for op in left_double:
            op_indices = find_end_indices_double_aug(cleaned, op)
            double_indices.extend(op_indices)
        left_indices = single_indices + double_indices
        left_indices.sort()
        result = []
        last_token = ''
        for i in range(len(cleaned)):
            char = cleaned[i]
            if len(result) == 0:
                result.append(char)
                last_token = char
                continue
            else:
                if char.isalpha() or is_number(char) or char == '(' or char in op_list:
                    if last_token.isalpha() or is_number(last_token) or last_token in left_token:
                        if not (((char.isalpha() or char in greek_letters) and (last_token.isalpha() or last_token in greek_letters)) or (char == '(' and (last_token.isalpha() or last_token in greek_letters))): #avoid x*y, partial * x and f*(x), alpha(x)
                            result.append('*')
                    elif last_token in ['}']:
                        if i-1 in left_indices:
                            result.append('*')
                    result.append(char)
                    last_token = char    
                else:
                    result.append(char)
                    last_token = char
        return result

op_need_correct = ['exp', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
              'arcsin', 'arccos', 'arctan', 'arccot', 'arcsec', 'arccsc',
              'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
              'arcsinh', 'arccosh', 'arctanh', 'arccoth', 'arcsech', 'arccsch']
simp_op_direct_correct = ['exp', 'arg', 'deg', 'det', 'dim', 'gcd', 'hom', 'ker', 'lg', 'ln', 'log', 'max', 'Pr']
trig_need_correct = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc']
arctrig_need_correct = ['arcsin', 'arccos', 'arctan', 'arccot', 'arcsec', 'arccsc', 'asin', 'acos', 'atan', 'acot', 'asec', 'acsc']
hyper_need_correct = ['sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch']
archyper_need_correct = ['arcsinh', 'arccosh', 'arctanh', 'arccoth', 'arcsech', 'arccsch', 'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch']

def correct_sin(input_string): #correct syntax mistakes like write sin instead of \\sin
    original = input_string
    for op in simp_op_direct_correct:
        corrected_string = re.sub(fr'(?<!\\)(?<!\w){op}', fr'\\{op} ', input_string)
        input_string = corrected_string
    corrected_string = re.sub(rf'(?<!\\)(?<!\w)min', fr'\\min ', input_string)
    input_string = corrected_string
    corrected_string = re.sub(rf'(?<!\\)(?<!lim\s)(?<!lim)(?<!\w)inf', fr'\\inf ', input_string)
    input_string = corrected_string
    corrected_string = re.sub(rf'(?<!\\)(?<!lim\s)(?<!lim)(?<!\w)sup', fr'\\sup ', input_string)
    input_string = corrected_string
    corrected_string = re.sub(rf'(?<!\\)(?<!\w)lim(?!\ssup)(?!sup)(?!\sinf)(?!inf)', fr'\\lim ', input_string)
    input_string = corrected_string
    corrected_string = re.sub(rf'(?<!\\)(?<!\w)liminf', fr'\\liminf ', input_string)
    input_string = corrected_string
    corrected_string = re.sub(rf'(?<!\\)(?<!\w)lim\s+inf', fr'\\liminf ', input_string)
    input_string = corrected_string
    corrected_string = re.sub(rf'(?<!\\)(?<!\w)limsup', fr'\\limsup ', input_string)
    input_string = corrected_string
    corrected_string = re.sub(rf'(?<!\\)(?<!\w)lim\s+sup', fr'\\limsup ', input_string)
    input_string = corrected_string
    corrected_string = re.sub(r'\.{2,}', r'\\ldots ', input_string)
    input_string = corrected_string   

    for op in trig_need_correct:
        pattern = rf'(?<!\\)(?<!\w)(?<!arc){op}(?!h)'
        corrected_string = re.sub(pattern, rf'\\{op} ', input_string)
        input_string = corrected_string
    for op in arctrig_need_correct:
        pattern = rf'(?<!\\)(?<!\w){op}(?!h)'
        corrected_string = re.sub(pattern, fr'\\{op} ', input_string)
        input_string = corrected_string
    for op in hyper_need_correct:
        pattern = rf'(?<!\\)(?<!\w)(?<!arc){op}'
        corrected_string = re.sub(pattern, fr'\\{op} ', input_string)
        input_string = corrected_string
    for op in archyper_need_correct:
        pattern = rf'(?<!\\)(?<!\w){op}'
        corrected_string = re.sub(pattern, fr'\\{op}'+'^{-1} ', input_string)
        input_string = corrected_string
    return corrected_string

def find_end_indices_one_aug(expression, op): #assistant function, find end index of operator with one augment
    stack = []
    indices = []
    i = 0
    while i < len(expression):
        if expression[i] == op:
            stack.append(i)
            i += 1
        elif expression[i] == '{' and stack:
            stack.append('{')
            i += 1
        elif expression[i] == '}' and stack:
            stack.pop()
            if stack and stack[-1] != '{':
                start_index = stack.pop()
                indices.append(i)
            i += 1
        else:
            i += 1
    return indices

def find_end_indices_double_aug(expression, op): #assistant function, find end index of operator with two augment
    stack = []
    indices = []
    i = 0
    while i < len(expression):
        if expression[i] == op:
            if expression[i+1] == '{' or expression[i+1] == '_':
                stack.append(i)
            i += 1
        elif expression[i] == '{' and stack:
            stack.append('{')
            i += 1
        elif expression[i] == '}' and stack:
            try:
                if expression[i:i+2] == ['}', '{']:
                    i += 2
                    continue
            except:
                pass
            stack.pop()
            if stack and stack[-1] != '{':
                start_index = stack.pop()
                indices.append(i)
            i += 1
        else:
            i += 1
    return indices

def find_text_indices(expression, op): #assistant function, find end index of text operators\
    stack = []
    indices = []
    i = 0
    while i < len(expression):
        if expression[i] == op:
            stack.append(i)
            i += 1
        elif expression[i] == '{' and stack and expression[i-1] == op:
            stack.append('{')
            i += 1
        elif expression[i] == '}' and stack:
            stack.pop()
            if stack and stack[-1] != '{':
                start_index = stack.pop()
                indices.append(start_index)
                indices.append(start_index+1)
                indices.append(i)
            i += 1
        else:
            i += 1
    return indices

aplhabet = ['\\alpha', '\\beta', '\\delta', '\\epsilon', '\\eta', '\\gamma', '\\kappa', '\\lambda', '\\mu', '\\omega', '\\phi', 
                    '\\pi', '\\psi', '\\rho', '\\sigma', '\\tau', '\\theta', '\\xi', '\\zeta', '\\varepsilon', '\\varphi', '\\Delta', 
                    '\\Gamma', '\\Lambda', '\\Omega', '\\Phi', '\\Pi', '\\Psi', '\\Sigma', '\\Theta']

def variables_unification(tokens): #optional: keep original or id or id1, id2
    vocab_dict = {}
    id = 0
    result = []
    for i in range(len(tokens)):
        if tokens[i].isalpha() or tokens[i] in aplhabet:
            if tokens[i] not in vocab_dict:
                vocab_dict[tokens[i]] = f"var_id{str(id)}"
                id += 1
                tokens[i] = vocab_dict[tokens[i]]
            else:
                tokens[i] = vocab_dict[tokens[i]]
    return tokens

##################### a DEMO to use this preprocess #########################
tok = LatexTokenizer()
formula = '\\frac{b}{c} \\sqrt{x}'
formula = correct_sin(formula.strip().strip('.'))
formula_processed = tok.standarization(formula)
print(formula_processed)
#############################################################################

