    if (@json(setting('feature_tts_azure', false)) == 1) {
        const selectedOptionsAzure = azureVoiceData[selectedLanguage];
        if (selectedOptionsAzure) {
            if (allowedAzureList.includes(selectedLanguage)) {
                selectedOptionsAzure.forEach(option => {
                    $("<option></option>")
                        .val(option.value)
                        .text(option.label + (" (Azure)"))
                        .attr('platform', "azure")
                        .attr('name', option.label + (" (Azure)"))
                        .appendTo(voiceSelect);
                });
            }
        } else {
            selectedOptionsAzure.forEach(option => {
                $("<option></option>")
                    .val(option.value)
                    .text(option.label + (" (Azure)"))
                    .attr('name', option.label + (" (Azure)"))
                    .attr('platform', "azure")
                    .appendTo(voiceSelect);
            });
        }
    }
