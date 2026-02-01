<?php

namespace App\Extensions\Chatbot\System\Http\Controllers;

use App\Domains\Engine\Enums\EngineEnum;
use App\Domains\Entity\Enums\EntityEnum;
use App\Extensions\Chatbot\System\Enums\EmbeddingTypeEnum;
use App\Extensions\Chatbot\System\Http\Requests\Train\DataRequest;
use App\Extensions\Chatbot\System\Http\Requests\Train\EmbedingRequest;
use App\Extensions\Chatbot\System\Http\Requests\Train\FileRequest;
use App\Extensions\Chatbot\System\Http\Requests\Train\QaRequest;
use App\Extensions\Chatbot\System\Http\Requests\Train\TextRequest;
use App\Extensions\Chatbot\System\Http\Requests\Train\TrainUrlRequest;
use App\Extensions\Chatbot\System\Http\Resources\Admin\ChatbotEmbeddingResource;
use App\Extensions\Chatbot\System\Models\Chatbot;
use App\Extensions\Chatbot\System\Models\ChatbotEmbedding;
use App\Extensions\Chatbot\System\Parsers\ExcelParser;
use App\Extensions\Chatbot\System\Parsers\LinkParser;
use App\Extensions\Chatbot\System\Parsers\PdfParser;
use App\Extensions\Chatbot\System\Parsers\TextParser;
use App\Extensions\Chatbot\System\Services\ChatbotService;
use App\Extensions\Chatbot\System\Services\OpenAI\EmbedingService;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Illuminate\Contracts\View\View;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Resources\Json\AnonymousResourceCollection;

class ChatbotTrainController extends Controller
{
    public function __construct(public ChatbotService $service) {}

    public function train(Chatbot $chatbot): View
    {
        return view('chatbot::train', [
            'chatbot' => $chatbot,
        ]);
    }

    public function trainData(DataRequest $request): AnonymousResourceCollection
    {
        return ChatbotEmbeddingResource::collection(
            $this->service->query()
                ->findOrFail($request->validated('id'))
                ->embeddings()
                ->when($request->validated('type'), fn ($query) => $query->where('type', $request->validated('type')))
                ->get()
        );
    }

    public function deleteEmbedding(EmbedingRequest $request): JsonResponse
    {
        $chatbot = $this->service->query()->findOrFail($request->validated('id'));

        $chatbot->embeddings()->whereIn('id', $request->validated('data'))->delete();

        return response()->json([
            'message' => 'Embedding deleted successfully',
            'status'  => 200,
        ]);
    }

    public function generateEmbedding(EmbedingRequest $request): JsonResponse|AnonymousResourceCollection
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'type'    => 'error',
                'message' => 'This feature is disabled in Demo version.',
            ], 403);
        }

        $chatbot = $this->service->query()->findOrFail($request->validated('id'));

        ini_set('max_execution_time', -1);

        $data = $request->validated('data');

        $embeddings = ChatbotEmbedding::query()
            ->whereNull('embedding')
            ->whereIn('id', $data)
            ->get();

        $aiEmbeddingModel = EntityEnum::TEXT_EMBEDDING_3_SMALL;

        if (! EntityEnum::from($chatbot->getAttribute('ai_embedding_model'))) {
            $chatbot->update([
                'ai_embedding_model' => EntityEnum::TEXT_EMBEDDING_3_SMALL->value,
            ]);

            $aiEmbeddingModel = EntityEnum::TEXT_EMBEDDING_3_SMALL;
        }

        foreach ($embeddings as $embedding) {
            $embeddingJson = app(EmbedingService::class)
                ->setChatbot($chatbot)
                ->setEntity($aiEmbeddingModel)
                ->generateEmbedding($embedding->getAttribute('content'));

            $embedding->update([
                'embedding'    => $embeddingJson->toArray(),
                'trained_at'   => now(),
            ]);
        }

        return ChatbotEmbeddingResource::collection($chatbot->embeddings()->get());
    }

    public function trainText(TextRequest $request): JsonResponse|AnonymousResourceCollection
    {

        if (Helper::appIsDemo()) {
            return response()->json([
                'type'    => 'error',
                'message' => 'This feature is disabled in Demo version.',
            ], 403);
        }

        $chatbot = $this->service->query()->findOrFail($request->validated('id'));

        ChatbotEmbedding::query()
            ->create([
                'type'       => EmbeddingTypeEnum::text,
                'chatbot_id' => $chatbot->getKey(),
                'url'        => null,
                'file'       => null,
                'engine'     => EngineEnum::OPEN_AI->value,
                'title'      => $request->validated('title'),
                'content'    => $request->validated('content'),
            ]);

        return ChatbotEmbeddingResource::collection(
            $chatbot->embeddings()
                ->wherenull('file')
                ->whereNull('url')->get()
        );
    }

    public function trainQa(QaRequest $request): JsonResponse|AnonymousResourceCollection
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'type'    => 'error',
                'message' => 'This feature is disabled in Demo version.',
            ], 403);
        }

        $chatbot = $this->service->query()->findOrFail($request->validated('id'));

        ChatbotEmbedding::query()
            ->create([
                'type'       => EmbeddingTypeEnum::qa,
                'chatbot_id' => $chatbot->getKey(),
                'url'        => null,
                'file'       => null,
                'engine'     => EngineEnum::OPEN_AI->value,
                'title'      => $request->validated('question'),
                'content'    => $request->validated('question') . ' : ' . $request->validated('answer'),
            ]);

        return ChatbotEmbeddingResource::collection(
            $chatbot->embeddings()
                ->wherenull('file')
                ->whereNull('url')->get()
        );
    }

    public function trainUrl(TrainUrlRequest $request): JsonResponse|AnonymousResourceCollection
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'type'    => 'error',
                'message' => 'This feature is disabled in Demo version.',
            ], 403);
        }

        $chatbot = $this->service->query()->findOrFail($request->validated('id'));

        $chatbot->setAttribute('engine', EngineEnum::OPEN_AI->value);

        app(LinkParser::class)
            ->setBaseUrl($request->validated('url'))
            ->crawl((bool) $request->validated('single'))
            ->insertEmbeddings($chatbot);

        return ChatbotEmbeddingResource::collection(
            $chatbot->embeddings()->whereNotNull('url')->get()
        );
    }

    public function trainFile(FileRequest $request)
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'type'    => 'error',
                'message' => 'This feature is disabled in Demo version.',
            ], 403);
        }

        $chatbot = $this->service->query()->findOrFail($request->validated('id'));

        /** @var \Illuminate\Http\UploadedFile $file */
        $file = $request->file('file');

        $extension = $file->getClientOriginalExtension();

        $defaultDisk = 'public';

        $path = $file->store('chatbot', ['disk' => $defaultDisk]);

        $name = $file->getClientOriginalName();

        $storagePath = config('filesystems.disks.' . $defaultDisk . '.root') . '/' . $path;

        $parser = match (true) {
            in_array($extension, ['xlsx', 'xls', 'csv']) => app(ExcelParser::class),
            in_array($extension, ['txt', 'json'])        => app(TextParser::class),
            default                                      => app(PdfParser::class),
        };

        $text = $parser->setPath($storagePath)->parse();

        ChatbotEmbedding::query()
            ->firstOrCreate([
                'type'       => EmbeddingTypeEnum::file,
                'chatbot_id' => $chatbot->getKey(),
                'url'        => null,
                'file'       => $path,
                'engine'     => EngineEnum::OPEN_AI->value,
            ], [
                'title'    => $name,
                'content'  => $text,
            ]);

        return ChatbotEmbeddingResource::collection(
            $chatbot->embeddings()->whereNotNull('file')->get()
        );
    }
}
