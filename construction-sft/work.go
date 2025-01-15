package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"io/ioutil"
	"os"
	"path/filepath"
	"text/template"
)

func main() {
	// Directory containing JSON files
	dir := "cache/values"
	outputFile := "rendered_prompts.json"
	templateFile := "prompt_template.tmpl"

	// Load the template
	tmpl, err := template.ParseFiles(templateFile)
	if err != nil {
		fmt.Println("Error loading template:", err)
		return
	}

	// Read all JSON files in the directory
	var renderedResults []string
	err = filepath.Walk(dir, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && filepath.Ext(path) == ".json" {
			// Parse the JSON file
			data, err := ioutil.ReadFile(path)
			if err != nil {
				return fmt.Errorf("error reading file %s: %w", path, err)
			}
			var jsonData map[string]interface{}
			if err := json.Unmarshal(data, &jsonData); err != nil {
				return fmt.Errorf("error parsing JSON from file %s: %w", path, err)
			}

			// Render the template with the parsed JSON data
			var renderedOutput bytes.Buffer
			if err := tmpl.Execute(&renderedOutput, jsonData); err != nil {
				return fmt.Errorf("error rendering template for file %s: %w", path, err)
			}
			renderedResults = append(renderedResults, renderedOutput.String())
		}
		return nil
	})

	if err != nil {
		fmt.Println("Error processing files:", err)
		return
	}

	// Open the output file
	output, err := os.Create(outputFile)
	if err != nil {
		fmt.Println("Error creating output file:", err)
		return
	}
	defer output.Close()

	// Create a JSON encoder and disable HTML escaping
	encoder := json.NewEncoder(output)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")

	// Write the rendered results to the output file
	if err := encoder.Encode(renderedResults); err != nil {
		fmt.Println("Error encoding rendered results:", err)
		return
	}

	fmt.Printf("Rendered prompts saved to %s\n", outputFile)
}