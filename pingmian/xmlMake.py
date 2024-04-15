import xml.dom.minidom as minidom
import time
outTime=time.strftime("%d/%m/%Y %H:%M", time.localtime())
dom = minidom.parse(r"\\wsl.localhost\debian\opt\shanghai\hour_settings_ETRS89-Run.xml")
root = dom.documentElement
names = root.getElementsByTagName('setoption')
lfusers=root.getElementsByTagName('lfuser')
for lfuser in lfusers:
    groups=lfuser.getElementsByTagName('group')
    for group in groups:
        textvars=group.getElementsByTagName('textvar')
    # textvars=lfuser.childNodes[0].getElementsByTagName('textvar')#"CalendarDayStart"
        for textvar in textvars:
            if textvar.getAttribute('name') == "StepStart":
                print(textvar.getAttribute('value'))
                textvar.setAttribute('value', '23/07/2021 01:00')
            elif textvar.getAttribute('name') == "StepEnd":
                print(textvar.getAttribute('value'))
                textvar.setAttribute('value', '23/07/2021 01:00')
for name in names:
    # 它的第一个子节点是一个textnode，存取的是真正的节点值
    if name.getAttribute('name')=="inflow":
        print(name.getAttribute('choice'))
        name.setAttribute('choice', str(0))

with open('default.xml', 'w', encoding='utf-8') as f:
    dom.writexml(f, addindent='\t', newl='\n', encoding='utf-8')
    # print(name.childNodes[0].nodeValue, end='\t')
    # if name.hasAttribute('age'):
    #     print(name.getAttribute('age'), end='\t')
    # print('')
