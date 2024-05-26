library(ggplot2)
library(dplyr)
library(rstudioapi)

current_path <- dirname(rstudioapi::getSourceEditorContext()$path)

file_path <- file.path(current_path, "bin", "Debug", "net8.0", "Host.csv")

data <- read.csv(file_path)

data <- data %>% filter(Tick >= 1)

stages <- c("Anfällige", "Exponierte", "Infizierte", "Genesene", "Verstorbene")

ggplot(data, aes(x = Tick, group = MyStage, colour = as.factor(MyStage))) +
  geom_line(stat = "count", size = 1.1) +  # Erhöht die Linienstärke
  scale_colour_manual(values = c("blue", "orange", "red", "green", "black"),
                      labels = stages,
                      name = "Stadium") +
  labs(x = "Zeit (Ticks)", y = "Anzahl der Agenten", colour = "dark") +
  theme_minimal()
