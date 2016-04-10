# Quick calculator to check the logic of your belief in something

def bayes(x, y, z):
	p = (x*y)/((x*y)+(z*(1.0-x)))
	percent = p * 100
	percent = int(percent)

print "Tell me about your hypothesis."
hy = raw_input("\nI think that: ")

print "\n Tell me about what evidence makes you think this."
ev = raw_input("\nI think that %s because: " % hy)

y = raw_input("\nWhat is the probability that %s if %s?: " % (ev, hy)) 
y = float(y)

z = raw_input("\nWhat is the probability that %s if it is not true that %s: " % (ev, hy))
z = float(z)

print "\nFinally, what is your prior probability? That is, what is the probability you would have given to the event happening if you were completely ignorant about whether %s or not." % ev
x = raw_input("\nEnter your 'prior probability' now: ")
x = float(x)

p = (x*y)/((x*y)+(z*(1.0-x)))

percent = p * 100
percent = int(percent)

print "\n\nThe probability of you thinking that %r being correct given %r is: %r%%." % (hy, ev, percent)
