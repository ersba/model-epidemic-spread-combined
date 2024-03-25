library(ggplot2)
library(dplyr)
# Modell ohne Tensor
data <- read.csv("/home/e7/RiderProjects/model-epidemic-spread-without-tensors/EpidemicSpreadWithoutTensors/bin/Debug/net6.0/Host.csv")

#Modell mit Tensor Combined
data <- read.csv("/home/e7/RiderProjects/model-epidemic-spread-combined/EpidemicSpreadCombined/bin/Debug/net6.0/Host.csv")

#Modell mit Tensor
data <- read.csv("/home/e7/RiderProjects/model-epidemic-spread/EpidemicSpread/bin/Debug/net6.0/Host.csv")

data <- data %>% filter(Tick >= 1)

stages <- c("Anfällige", "Exponierte", "Infizierte", "Genesene", "Verstorbene")

ggplot(data, aes(x = Tick, group = MyStage, colour = as.factor(MyStage))) +
  geom_line(stat = "count", size = 1.1) +  # Erhöht die Linienstärke
  scale_colour_manual(values = c("blue", "orange", "red", "green", "black"),
                      labels = stages,
                      name = "Stadium") +
  labs(x = "Zeit (Ticks)", y = "Anzahl der Agenten", colour = "dark") +
  theme_minimal()
