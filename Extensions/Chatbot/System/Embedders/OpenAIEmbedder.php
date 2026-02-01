<?php

namespace App\Extensions\Chatbot\System\Embedders;

use App\Domains\Entity\Facades\Entity;
use App\Extensions\Chatbot\System\Embedders\Contracts\Embedder;
use App\Helpers\Classes\ApiHelper;
use Exception;
use Illuminate\Validation\ValidationException;
use OpenAI;
use OpenAI\Responses\Embeddings\CreateResponse;

class OpenAIEmbedder extends Embedder
{
    public function generate(): CreateResponse
    {
        $arrayInput = $this->getInput();

        if (is_array($arrayInput)) {
            $input = implode(',', $arrayInput);
        } else {
            $input = is_string($arrayInput) ? $arrayInput : '';
        }

        $driver = Entity::driver($this->getEntity())
            ->forUser($this->getUser())
            ->input($input)
            ->calculateCredit();

        try {
            $driver->redirectIfNoCreditBalance();
        } catch (Exception $e) {
            throw ValidationException::withMessages(['message' => $e->getMessage()]);
        }

        $driver->decreaseCredit();

        return $this->client()->embeddings()->create([
            'model' => $this->getEntity()->value,
            'input' => $this->getInput(),
        ]);
    }

    public function client(): OpenAI\Client
    {
        return OpenAI::client(ApiHelper::setOpenAiKey());
    }
}
