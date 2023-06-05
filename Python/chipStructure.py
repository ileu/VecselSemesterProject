# import matplotlib.pyplot as plt
# import numpy as np

# x = np.arange(0, 20, 0.5)

# height = 0.75
# plt.fill_between([0, height], [1, 1], color="gray")  # darkgoldenrod or gray

# # plt.fill_between([height, height + 0.5], [1, 1], color="darkgoldenrod") # darkgoldenrod or gray
# # height = height + 0.5

# for i in range(5):
#     plt.fill_between([height, height + 0.3], [0.6], color="lightsteelblue")
#     plt.fill_between([height + 0.3, height + 0.6], [0.5], color="dodgerblue")
#     height = height + 0.6

# # plt.fill_between([height, height + 2], [0.7], color="darkgreen")
# # plt.fill_between([height + 1, height + 1.2], [0.9], color="red", label="Absorber qw")
# # height = height + 2

# for i in range(5):
#     plt.fill_between([height, height + 0.5], [0.6], color="lightsteelblue")
#     plt.fill_between([height + 0.5, height + 1], [0.5], color="dodgerblue")
#     height = height + 1

# plt.fill_between([height, height + 3], [0.7], color="darkgreen")
# for i in range(5):
#     plt.fill_between(
#         [height + 0.5 * (i + 1), height + 0.5 * (i + 1) + 0.1], [0.9], color="orange", label="Gain QW"
#     )

# height = height + 3
# plt.fill_between([height, height + 1.5], [0.4], color="darkorchid", label="AR coating")
# plt.xlim(0, 15)

# frame1 = plt.gca()
# frame1.axes.get_xaxis().set_visible(False)
# frame1.axes.get_yaxis().set_visible(False)

# plt.tight_layout()
# plt.savefig(r"Images\SVA167-b2.png")
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,20,2000)

p = 0.005*(15*x**2-x**3)
plt.plot(x,p)
plt.show()
