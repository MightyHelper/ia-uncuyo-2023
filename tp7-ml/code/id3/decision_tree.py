import math
from typing import Any, Union
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Example Input Attributes Goal
# Alt Bar Fri Hun Pat Price Rain Res Type Est WillWait
# x1 Yes No No Yes Some $$$ No Yes French 0–10 y1 = Yes
# x2 Yes No No Yes Full $ No No Thai 30–60 y2 = No
# x3 No Yes No No Some $ No No Burger 0–10 y3 = Yes
# x4 Yes No Yes Yes Full $ Yes No Thai 10–30 y4 = Yes
# x5 Yes No Yes No Full $$$ No Yes French >60 y5 = No
# x6 No Yes No Yes Some $$ Yes Yes Italian 0–10 y6 = Yes
# x7 No Yes No No None $ Yes No Burger 0–10 y7 = No
# x8 No No No Yes Some $$ Yes Yes Thai 0–10 y8 = Yes
# x9 No Yes Yes No Full $ Yes No Burger >60 y9 = No
# x10 Yes Yes Yes Yes Full $$$ No Yes Italian 10–30 y10 = No
# x11 No No No No None $ No No Thai 0–10 y11 = No
# x12 Yes Yes Yes Yes Full $ No No Burger 30–60 y12 = Yes
dataset = [
	{'alt': True, 'bar': False, 'fri': False, 'hun': True, 'pat': 'Some', 'price': '$$$', 'rain': False, 'res': True, 'type': 'French', 'est': '0-10', 'will_wait': True},
	{'alt': True, 'bar': False, 'fri': False, 'hun': True, 'pat': 'Full', 'price': '$', 'rain': False, 'res': False, 'type': 'Thai', 'est': '30-60', 'will_wait': False},
	{'alt': False, 'bar': True, 'fri': False, 'hun': False, 'pat': 'Some', 'price': '$', 'rain': False, 'res': False, 'type': 'Burger', 'est': '0-10', 'will_wait': True},
	{'alt': True, 'bar': False, 'fri': True, 'hun': True, 'pat': 'Full', 'price': '$', 'rain': True, 'res': True, 'type': 'Thai', 'est': '10-30', 'will_wait': True},
	{'alt': True, 'bar': False, 'fri': True, 'hun': False, 'pat': 'Full', 'price': '$$$', 'rain': False, 'res': True, 'type': 'French', 'est': '>60', 'will_wait': False},
	{'alt': False, 'bar': True, 'fri': False, 'hun': True, 'pat': 'Some', 'price': '$$', 'rain': True, 'res': True, 'type': 'Italian', 'est': '0-10', 'will_wait': True},
	{'alt': False, 'bar': True, 'fri': False, 'hun': False, 'pat': 'None', 'price': '$', 'rain': True, 'res': True, 'type': 'Burger', 'est': '0-10', 'will_wait': False},
	{'alt': False, 'bar': False, 'fri': False, 'hun': True, 'pat': 'Some', 'price': '$$', 'rain': True, 'res': True, 'type': 'Thai', 'est': '0-10', 'will_wait': True},
	{'alt': False, 'bar': True, 'fri': True, 'hun': False, 'pat': 'Full', 'price': '$', 'rain': True, 'res': False, 'type': 'Burger', 'est': '>60', 'will_wait': False},
	{'alt': True, 'bar': True, 'fri': True, 'hun': True, 'pat': 'Full', 'price': '$$$', 'rain': False, 'res': True, 'type': 'Italian', 'est': '10-30', 'will_wait': False},
	{'alt': False, 'bar': False, 'fri': False, 'hun': False, 'pat': 'None', 'price': '$', 'rain': False, 'res': False, 'type': 'Thai', 'est': '0-10', 'will_wait': False},
	{'alt': True, 'bar': True, 'fri': True, 'hun': True, 'pat': 'Full', 'price': '$', 'rain': False, 'res': False, 'type': 'Burger', 'est': '30-60', 'will_wait': True}
]

# outlook,temp,humidity,windy,play
# sunny,hot,high,false,no
# sunny,hot,high,true,no
# overcast,hot,high,false,yes
# rainy,mild,high,false,yes
# rainy,cool,normal,false,yes
# rainy,cool,normal,true,no
# overcast,cool,normal,true,yes
# sunny,mild,high,false,no
# sunny,cool,normal,false,yes
# rainy,mild,normal,false,yes
# sunny,mild,normal,true,yes
# overcast,mild,high,true,yes
# overcast,hot,normal,false,yes
# rainy,mild,high,true,no
tennis = [
	{'outlook': 'sunny', 'temp': 'hot', 'humidity': 'high', 'windy': False, 'play': False},
	{'outlook': 'sunny', 'temp': 'hot', 'humidity': 'high', 'windy': True, 'play': False},
	{'outlook': 'overcast', 'temp': 'hot', 'humidity': 'high', 'windy': False, 'play': True},
	{'outlook': 'rainy', 'temp': 'mild', 'humidity': 'high', 'windy': False, 'play': True},
	{'outlook': 'rainy', 'temp': 'cool', 'humidity': 'normal', 'windy': False, 'play': True},
	{'outlook': 'rainy', 'temp': 'cool', 'humidity': 'normal', 'windy': True, 'play': False},
	{'outlook': 'overcast', 'temp': 'cool', 'humidity': 'normal', 'windy': True, 'play': True},
	{'outlook': 'sunny', 'temp': 'mild', 'humidity': 'high', 'windy': False, 'play': False},
	{'outlook': 'sunny', 'temp': 'cool', 'humidity': 'normal', 'windy': False, 'play': True},
	{'outlook': 'rainy', 'temp': 'mild', 'humidity': 'normal', 'windy': False, 'play': True},
	{'outlook': 'sunny', 'temp': 'mild', 'humidity': 'normal', 'windy': True, 'play': True},
	{'outlook': 'overcast', 'temp': 'mild', 'humidity': 'high', 'windy': True, 'play': True},
	{'outlook': 'overcast', 'temp': 'hot', 'humidity': 'normal', 'windy': False, 'play': True},
	{'outlook': 'rainy', 'temp': 'mild', 'humidity': 'high', 'windy': True, 'play': False}
]


def plurality_value(examples: list[dict[str, Any]], y: str) -> Any:
	# return the most common classification
	examples: list[int] = [1 if example[y] else 0 for example in examples]
	total_examples: int = len(examples)
	positive_examples: int = sum(examples)
	negative_examples: int = total_examples - positive_examples
	return positive_examples > negative_examples


def entropy(examples: list[dict[str, Any]], y: str):
	examples: list[int] = [1 if example[y] else 0 for example in examples]
	total_examples: int = len(examples)
	positive_examples: int = sum(examples)
	p: float = positive_examples / total_examples
	return calculate_boolean_entropy(p)


def calculate_boolean_entropy(p: float) -> float:
	ent: float = 0.0
	q: float = 1 - p
	if p != 0: ent -= p * math.log2(p)
	if q != 0: ent -= q * math.log2(q)
	return ent


def remainder(attribute: str, examples: list[dict[str, Any]], y: str):
	remain: float = 0.0
	for value in set([example[attribute] for example in examples]):
		branch_examples = [example for example in examples if example[attribute] == value]
		remain += (len(branch_examples) / len(examples)) * entropy(branch_examples, y)
	return remain


def importance(attribute: str, examples: list[dict[str, Any]], y: str):
	# return the importance of attribute
	# importance = entropy(examples) - remainder(attribute, examples)
	return entropy(examples, y) - remainder(attribute, examples, y)


def run_tests():
	y = "will_wait"
	assert abs(calculate_boolean_entropy(0.01) - 0.08) < 0.001
	assert abs(importance("pat", dataset, y) - 0.54) < 0.001
	assert abs(importance("type", dataset, y)) < 0.001
	table = Table(title="Column importance")
	table.add_column("Attribute")
	table.add_column("Importance")
	for attribute in dataset[0].keys():
		table.add_row(attribute, str(importance(attribute, dataset, y)))
	console = Console()
	console.print(table)


def decision_tree_learning(examples: list[dict[str, Any]], attributes: set[str], parent_examples, y) -> Union[dict[str, Any], Any]:
	# return a tree
	pickable_attributes = [attr for attr in attributes if attr != y]
	if len(examples) == 0:
		return plurality_value(parent_examples, y)
	elif len({e[y] for e in examples}) == 1:
		return examples[0][y]
	elif len(pickable_attributes) == 0:
		return plurality_value(examples, y)
	else:
		a = max(pickable_attributes, key=lambda attribute: importance(attribute, examples, y))
		tree = {a: {}}
		for value in set([example[a] for example in examples]):
			exs = [example for example in examples if example[a] == value]
			subtree = decision_tree_learning(exs, {attr for attr in attributes if attr != a}, examples, y)
			tree[a][value] = subtree
		return tree


def classify(tree: Union[dict[str, Any], bool], example: dict[str, Any]):
	# return the classification of example
	if isinstance(tree, bool):
		return tree
	else:
		attribute = list(tree.keys())[0]
		value = example[attribute]
		return classify(tree[attribute][value], example)


def main(dataset_to_use, y):
	# run_tests()
	valid_attributes = set(dataset_to_use[0].keys())
	valid_attributes = [*[at for at in valid_attributes if at != y], y]
	trained_tree = decision_tree_learning(dataset_to_use, set(valid_attributes), [], y)
	rprint(trained_tree)
	table = Table(title="Decision Tree")
	for v in valid_attributes:
		table.add_column(v)
	table.add_column("Predicted")
	table.add_column("Result")
	for example in dataset_to_use:
		row = []
		for v in valid_attributes:
			row.append(str(example[v]))
		result = classify(trained_tree, example)
		row.append(str(result))
		row.append("OK" if result == example[y] else "ERROR")
		table.add_row(*row)
	console = Console()
	console.print(table)


if __name__ == '__main__':
	main(dataset, "will_wait")
	main(tennis, "play")
