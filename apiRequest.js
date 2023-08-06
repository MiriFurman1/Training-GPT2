async function query(data) {
	const response = await fetch(
		"https://api-inference.huggingface.co/models/MiriFur/gpt2-recipes",
		{
			headers: { Authorization: "Bearer " },
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.json();
	return result;
}

query({"inputs": "generate one random recipe containing cereal. A recipe contains title, categories, servings, ingredients,  directions. ",
"parameters": {"max_length": 499}}).then((response) => {
	console.log(JSON.stringify(response));
});