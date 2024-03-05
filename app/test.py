from src.complaints_analyser import ComplaintsParser


parser = ComplaintsParser()
example_complaint = """
I have spoke with wellsfargo about overdraft protection, the stated that I had overdraft protection on my account, 
they are allowing money to come from my account when there is nothing there, but I have been hit XXXX times in XXXX day 
with overdraft fees. The most recent date has been XXXXXXXX XXXX
"""

output = parser.analyse_text(example_complaint)

breakpoint()