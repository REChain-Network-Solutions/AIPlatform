<?php

namespace App\Extensions\AzureTTS\System\Services;

class AzureService
{
    private $azure_api_key;

    private $azure_region;

    private $endpoint = '.tts.speech.microsoft.com/cognitiveservices/v1';

    public function __construct()
    {
        $this->azure_api_key = setting('azure_api_key', '');
        $this->azure_region = setting('azure_region', '');
    }

    public function synthesizeSpeech($voice_id, $input_text, $lang)
    {
        $url = 'https://' . $this->azure_region . $this->endpoint;
        $input_text = e($input_text);
        $ssml =
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
		xmlns:mstts="http://www.w3.org/2001/mstts"
		xmlns:emo="http://www.w3.org/2009/10/emotionml"
		xml:lang="' . $lang . '"><voice name="' . $voice_id . '">' . $input_text . '</voice></speak>';

        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_CUSTOMREQUEST, 'POST');
        curl_setopt($ch, CURLOPT_HTTPHEADER, [
            'Ocp-Apim-Subscription-Key: ' . $this->azure_api_key,
            'Content-Type: application/ssml+xml',
            'X-Microsoft-OutputFormat: audio-24khz-48kbitrate-mono-mp3',
            'User-Agent: curl',
        ]);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $ssml);
        $audio = curl_exec($ch);
        if (curl_errno($ch)) {
            $error = json_encode(['error' => 'Error: ' . curl_error($ch)]);

            return response()->json($error, 500);
        }
        curl_close($ch);

        return $audio;
    }
}
