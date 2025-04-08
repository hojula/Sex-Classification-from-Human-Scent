class Statistics:
    def __init__(self):
        self.best_validation_matrix = None
        self.best_id_correct_incorrect_validation = None
        self.best_validation_error = 1
        self.best_validation_loss = None
        self.validation_matrix = None
        self.id_correct_incorrect_validation = None
        self.validation_error = 1
        self.validation_loss = None

    def init_new_epoch(self):
        self.validation_matrix = None
        self.id_correct_incorrect_validation = {}
        self.validation_error = 1
        self.validation_loss = None

    def update_best(self):
        if self.validation_error < self.best_validation_error:
            self.best_validation_error = self.validation_error
            self.best_validation_matrix = self.validation_matrix
            self.best_id_correct_incorrect_validation = self.id_correct_incorrect_validation
            self.best_validation_loss = self.validation_loss

    def sort_by_key(self, dictionary):
        def key_func(k):
            letter = k[0]
            number = int(k[1:])
            return (letter, number)

        return {k: dictionary[k] for k in sorted(dictionary, key=key_func)}

    def __str__(self):
        self.best_id_correct_incorrect_validation = self.sort_by_key(self.best_id_correct_incorrect_validation)
        print("-----------------")
        out = "Best validation error: " + "{:.2f}".format(self.best_validation_error) + "\n"
        out += "Best validation loss: " + "{:.2f}".format(self.best_validation_loss) + "\n"
        out += "Men: Correct: " + str(self.best_validation_matrix["M"]["Correct"]) + " Incorrect: " + \
               str(self.best_validation_matrix["M"]["Incorrect"]) + "\n"
        out += "Women: Correct: " + str(self.best_validation_matrix["F"]["Correct"]) + " Incorrect: " + \
               str(self.best_validation_matrix["F"]["Incorrect"]) + "\n"
        for key in self.best_id_correct_incorrect_validation:
            out += "ID: " + key + " Correct: " + str(self.best_id_correct_incorrect_validation[key][
                                                         "Correct"]) + " Incorrect: " + str(
                self.best_id_correct_incorrect_validation[key]["Incorrect"]) + "\n"
        out += "-----------------"
        return out

    def update_correct_incorrect(self, who, id_correct_incorrect, correct):
        if correct:
            if who not in id_correct_incorrect:
                id_correct_incorrect[who] = {"Correct": 0, "Incorrect": 0}
            id_correct_incorrect[who]["Correct"] += 1
        else:
            if who not in id_correct_incorrect:
                id_correct_incorrect[who] = {"Correct": 0, "Incorrect": 0}
            id_correct_incorrect[who]["Incorrect"] += 1

    def count_statistic(self, predicted, true, ids):
        validation_matrix = {"M": {"Correct": 0, "Incorrect": 0}, "F": {"Correct": 0, "Incorrect": 0}}
        for j in range(len(predicted)):
            if predicted[j] == 1:
                if true[j] == 1:
                    self.update_correct_incorrect(ids[j], self.id_correct_incorrect_validation, True)
                    validation_matrix["M"]["Correct"] += 1
                else:
                    self.update_correct_incorrect(ids[j], self.id_correct_incorrect_validation, False)
                    validation_matrix["F"]["Incorrect"] += 1
            else:
                if true[j] == 0:
                    self.update_correct_incorrect(ids[j], self.id_correct_incorrect_validation, True)
                    validation_matrix["F"]["Correct"] += 1
                else:
                    self.update_correct_incorrect(ids[j], self.id_correct_incorrect_validation, False)
                    validation_matrix["M"]["Incorrect"] += 1
        if self.validation_matrix is None:
            self.validation_matrix = validation_matrix
        else:
            self.validation_matrix["M"]["Correct"] += validation_matrix["M"]["Correct"]
            self.validation_matrix["M"]["Incorrect"] += validation_matrix["M"]["Incorrect"]
            self.validation_matrix["F"]["Correct"] += validation_matrix["F"]["Correct"]
            self.validation_matrix["F"]["Incorrect"] += validation_matrix["F"]["Incorrect"]
