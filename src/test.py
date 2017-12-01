class colors:
    red = '\033[91m'
    green = '\033[92m'
    yellow = '\033[93m'
    blue = '\033[94m'
    purple = '\033[95m'
    teal = '\033[96m'
    grey = '\033[97m'
    black = '\033[98m'
    #black = '\033[99m'
    
    end = '\033[0m'
    bold = '\033[1m'
    under = '\033[4m'

if __name__ == "__main__":
	print(colors.blue + colors.bold + "Does not work in my console :/" + colors.end)
