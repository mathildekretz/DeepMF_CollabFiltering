# Deep Matrix Factorization for Collaborative Filtering - DataLabAssignement1

## IASD Master Program 2023/2024 - PSL Research University

### About this project

This project is the first homework assignment for the Data Science Lab class of the IASD (Artificial Intelligence, Systems and Data) Master Program 2023/2024, at PSL Research University (Université PSL).

*The project successfully addressed the following objectives:*
- Enhanced collaborative filtering through matrix factorization employing various gradient descent methods, providing a comprehensive exploration of their impact on the factorization process.
- Introduced a novel approach named DeepMF (Deep Matrix Factorization), innovatively transposing the principles of deep learning into matrix factorization. This technique operates without neural networks, utilizing recursive factorization of successive error matrices to adapt its primal optimization strategy.
- Conducted a detailed analysis and implementation of DeepMF, demonstrating its effectiveness in dynamically adapting to evolving error structures and showcasing its potential advantages over traditional matrix factorization techniques.

## General information

The report can be viewed in the [report.pdf](report.pdf) file. It answers to the instructions given in the [assignment_1_slides_instructions.pdf](assignment_1_slides_instructions.pdf) file provided by the professors.

The rest of the instructions can be found below. If you want to copy and recreate this project, or test it for yourself, some important information to know.

**generate.py**
Use the file *generate.py* to complete your ratings table. 
It takes in argument *--name* the name of the files you want to use, and it saves the complete matrix as *output.npy*.
DO NOT CHANGE THE LINES TO LOAD AND SAVE THE TABLE. Between those two you are free to use any method for matrix completion. 
Example:
  > python3 generate.py --name ratings_train.npy

**requirements.txt**
Among the good practices of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different libraries you might use. 
When your code will be tested, we will execute: 
  > pip install -r requirements.txt

**ratings_train.npy** and **ratings_test.npy**
These two files contain the sparse rating matrices to factorize. They are extracted from the MovieLens dataset, and modified by the professors for this assignment.
Note that the original sparse matrix is a sum of the train and test matrices.

**namesngenre.npy**
This file contains additional information on the dataset, extracted from the MovieLens dataset.

---

### Acknowledgement

This project was made possible with the guidance and support of the following :

- **Prof. Benjamin Negrevergne**
  - Professor at *Université Paris-Dauphine, PSL*
  - Researcher in the *MILES Team* at *LAMSADE, UMR 7243* at *Université Paris-Dauphine, PSL* and *Université PSL*
  - Co-director of the IASD Master Program with Olivier Cappé

- **Alexandre Vérine**
  - PhD candidate at *LAMSADE, UMR 7243* at *Université Paris-Dauphine, PSL* and *Université PSL*
 
This project was a group project, and was made possible thanks to the collaboration of :

- **Mathilde Kretz**, *IASD Master Program 2023/2024 student, at PSL Research University*
- **Thomas Boudras**, *IASD Master Program 2023/2024 student, at PSL Research University*
- **Alexandre Ngau**, *IASD Master Program 2023/2024 student, at PSL Research University*

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

**Note:**

This project is part of ongoing research and is subject to certain restrictions. Please refer to the license section and the [LICENSE.md](LICENSE.md) file for details on how to use this code for research purposes.
