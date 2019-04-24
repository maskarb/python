import math


class UnivariateDensityDerivative(object):

    P_UL = 500
    R = 1

    def __init__(
        self, NSources, MTargets, pSources, pTargets, Bandwidth, Order, epsilon
    ):
        self.N = NSources
        self.M = MTargets
        self.px = pSources
        self.py = pTargets
        self.pD = [None] * MTargets
        self.h = Bandwidth
        self.r = Order
        self.eps = epsilon

        self.h_square = self.h ** 2
        self.two_h_square = 2 * self.h_square
        self.q = math.pow(-1, self.r) / (
            math.sqrt(2 * math.pi) * self.N * math.pow(self.h, self.r + 1)
        )

        self.choose_parameters()
        self.space_sub_division()
        self.compute_a()
        self.compute_b()

    def choose_parameters(self):
        self.rx = self.h / 2

        self.K = math.ceil(1 / self.rx)
        self.rx = 1 / self.K
        rx_square = self.rx ** 2

        r_term = math.sqrt(math.factorial(self.r))

        rr = min(self.R, 2 * self.h * math.sqrt(math.log(r_term / self.eps)))

        self.ry = self.rx + rr

        self.p = 0
        error = 1
        temp = 1
        comp_eps = self.eps / r_term

        while (error > comp_eps) and (self.p <= self.P_UL):
            self.p += 1
            b = min(
                ((self.rx + math.sqrt((rx_square) + (8 * self.p * self.h_square))) / 2),
                self.ry,
            )
            c = self.rx - b
            temp = temp * (((self.rx * b) / self.h_square) / self.p)
            error = temp * (math.exp(-(c ** 2) / 2 * self.two_h_square))
        self.p += 1

    def space_sub_division(self):
        self.pClusterCenter = [None] * self.K
        for i in range(self.K):
            self.pClusterCenter[i] = (i * self.rx) + (self.rx / 2)

        self.pClusterIndex = [None] * self.N
        for i in range(self.N):
            self.pClusterIndex[i] = min(math.floor(self.px[i] / self.rx), self.K - 1)

    def compute_a(self):
        r_factorial = math.factorial(self.r)

        l_constant = [None] * (math.floor(self.r / 2) + 1)
        l_constant[0] = 1
        for l in range(1, math.floor(self.r / 2) + 1):
            l_constant[l] = l_constant[l - 1] * (-1 / (2 * l))

        m_constant = [None] * (self.r + 1)
        m_constant[0] = 1
        for m in range(1, self.r + 1):
            m_constant[m] = m_constant[m - 1] * (-1 / m)

        num_of_a_terms = len(l_constant) * len(m_constant)

        k = 0
        self.a_terms = [None] * num_of_a_terms
        for l in range(math.floor(self.r / 2) + 1):
            for m in range(self.r - (2 * l) + 1):
                self.a_terms[k] = (
                    l_constant[l]
                    * m_constant[m]
                    * r_factorial
                    / math.factorial(self.r - (2 * l) - m)
                )
                k += 1

    def compute_b(self):
        num_of_b_terms = self.K * self.p * (self.r + 1)
        self.b_terms = [0] * num_of_b_terms

        k_factorial = [None] * self.p
        k_factorial[0] = 1
        for i in range(1, self.p):
            k_factorial[i] = k_factorial[i - 1] / i

        temp3 = [None] * (self.p + self.r)

        for i in range(self.N):
            cluster_number = self.pClusterIndex[i]
            temp1 = (self.px[i] - self.pClusterCenter[cluster_number]) / self.h
            temp2 = math.exp(-temp1 * temp1 / 2)
            temp3[0] = 1
            for k in range(1, self.p + self.r):
                temp3[k] = temp3[k - 1] * temp1
            for k in range(self.p):
                for m in range(self.r + 1):
                    self.b_terms[
                        (cluster_number * self.p * (self.r + 1))
                        + ((self.r + 1) * k)
                        + m
                    ] += (temp2 * temp3[k + m])

        for n in range(self.K):
            for k in range(self.p):
                for m in range(self.r + 1):
                    self.b_terms[
                        (n * self.p * (self.r + 1)) + ((self.r + 1) * k) + m
                    ] *= (k_factorial[k] * self.q)

    def evaluate(self):
        temp3 = [0] * (self.p + self.r)

        for j in range(self.M):
            self.pD[j] = 0

            target_cluster_number = min(math.floor(self.py[j] / self.rx), self.K - 1)
            temp1 = self.py[j] - self.pClusterCenter[target_cluster_number]
            dist = abs(temp1)
            while (
                (dist <= self.ry)
                and (target_cluster_number < self.K)
                and (target_cluster_number >= 0)
            ):
                temp1 = self.py[j] - self.pClusterCenter[target_cluster_number]
                dist = abs(temp1)
                temp2 = math.exp(-temp1 * temp1 / self.two_h_square)
                temp1h = temp1 / self.h
                temp3[0] = 1
                for i in range(1, self.p + self.r):
                    temp3[i] = temp3[i - 1] * temp1h

                for k in range(self.p):
                    dummy = 0
                    for l in range(math.floor(self.r / 2) + 1):
                        for m in range(self.r - 2 * l + 1):
                            self.pD[j] = self.pD[j] + (
                                self.a_terms[dummy]
                                * self.b_terms[
                                    (target_cluster_number * self.p * (self.r + 1))
                                    + ((self.r + 1) * k)
                                    + m
                                ]
                                * temp2
                                * temp3[k + self.r - (2 * l) - m]
                            )
                            dummy += 1
                target_cluster_number += 1

            target_cluster_number = (
                min(math.floor(self.py[j] / self.rx), self.K - 1) - 1
            )
            if target_cluster_number >= 0:
                temp1 = self.py[j] - self.pClusterCenter[target_cluster_number]
                dist = abs(temp1)

                while (
                    (dist <= self.ry)
                    and (target_cluster_number < self.K)
                    and (target_cluster_number >= 0)
                ):
                    temp1 = self.py[j] - self.pClusterCenter[target_cluster_number]
                    dist = abs(temp1)
                    temp2 = math.exp(-temp1 * temp1 / self.two_h_square)
                    temp1h = temp1 / self.h
                    temp3[0] = 1
                    for i in range(1, self.p + self.r):
                        temp3[i] = temp3[i - 1] * temp1h

                    for k in range(self.p):
                        dummy = 0
                        for l in range(math.floor(self.r / 2) + 1):
                            for m in range(self.r - 2 * l + 1):
                                self.pD[j] = self.pD[j] + (
                                    self.a_terms[dummy]
                                    * self.b_terms[
                                        (target_cluster_number * self.p * (self.r + 1))
                                        + ((self.r + 1) * k)
                                        + m
                                    ]
                                    * temp2
                                    * temp3[k + self.r - (2 * l) - m]
                                )
                                dummy += 1
                    target_cluster_number -= 1
