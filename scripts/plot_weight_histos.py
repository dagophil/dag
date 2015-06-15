import sys
import numpy
import matplotlib.pyplot as plt


def main():

    with open("beta.txt") as f:
        betas = [float(l.strip()) for l in f]

    betas = numpy.array(betas)
    hist, bin_edges = numpy.histogram(betas, bins=100)
    bin_width = 1 * (bin_edges[1] - bin_edges[0])
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(centers, hist, align="center", width=bin_width)
    plt.title("Leaf weight distribution")
    plt.xlabel("weight")
    plt.ylabel("number of leaves")
    plt.show()

    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)
