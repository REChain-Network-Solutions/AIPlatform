<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\DB;

return new class extends Migration
{
    /**
     * These values must remain independent of application enums because the web
     * updater runs migrations in the request that loaded the previous release.
     *
     * @var array<int, array{key: string, title: string, engine: string, token_type: string, logo: string}>
     */
    private array $entities = [
        [
            'key'        => 'gpt-6-astra',
            'title'      => "GPT-6 Astra (OpenAI's most capable model, 1M+ context, reasoning with low/medium/high/xhigh/max effort, vision, computer use.)",
            'engine'     => 'openai',
            'token_type' => 'word',
            'logo'       => 'upload/enginelogo/openai.svg',
        ],
        [
            'key'        => 'lyria-3.5',
            'title'      => 'Google Lyria 3.5 for AI Music Pro (Full-length song generation, improved vocals and lyrics)',
            'engine'     => 'gemini',
            'token_type' => 'second',
            'logo'       => 'upload/enginelogo/gemini_logo.svg',
        ],
        [
            'key'        => 'gemini-3.8-flash',
            'title'      => 'Gemini 3.8 Flash Enhanced agentic reasoning and coding, 1M token context, multimodal input',
            'engine'     => 'gemini',
            'token_type' => 'word',
            'logo'       => 'upload/enginelogo/gemini_logo.svg',
        ],
        [
            'key'        => 'gpt-image-2.5-sunburst',
            'title'      => 'GPT-IMAGE-2.5 Sunburst Most capable image model for generation and editing with precision control.',
            'engine'     => 'openai',
            'token_type' => 'image',
            'logo'       => 'upload/enginelogo/openai.svg',
        ],
        [
            'key'        => 'gpt-image-2.5-flare',
            'title'      => 'GPT-IMAGE-2.5 Flare Fast, high-quality everyday image generation.',
            'engine'     => 'openai',
            'token_type' => 'image',
            'logo'       => 'upload/enginelogo/openai.svg',
        ],
    ];

    public function up(): void
    {
        foreach ($this->entities as $entity) {
            $entityId = $this->upsertEntity($entity);

            $this->ensureToken($entityId, $entity['token_type']);
        }

        Cache::forget('entities');
    }

    public function down(): void
    {
        foreach ($this->entities as $entity) {
            DB::table('entities')
                ->where('key', $entity['key'])
                ->where('image', $entity['logo'])
                ->update([
                    'image'      => null,
                    'updated_at' => now(),
                ]);
        }

        Cache::forget('entities');
    }

    /**
     * @param  array{key: string, title: string, engine: string, token_type: string, logo: string}  $entity
     */
    private function upsertEntity(array $entity): int
    {
        $entityId = DB::table('entities')->where('key', $entity['key'])->value('id');
        $attributes = [
            'engine'     => $entity['engine'],
            'title'      => $entity['title'],
            'status'     => 'enabled',
            'image'      => $entity['logo'],
            'updated_at' => now(),
        ];

        if ($entityId !== null) {
            DB::table('entities')->where('id', $entityId)->update($attributes);

            return (int) $entityId;
        }

        return DB::table('entities')->insertGetId([
            'key'        => $entity['key'],
            'created_at' => now(),
            ...$attributes,
        ]);
    }

    private function ensureToken(int $entityId, string $tokenType): void
    {
        $tokenId = DB::table('tokens')
            ->where('entity_id', $entityId)
            ->where('type', $tokenType)
            ->value('id');

        if ($tokenId !== null) {
            DB::table('tokens')->where('id', $tokenId)->update(['updated_at' => now()]);

            return;
        }

        DB::table('tokens')->insert([
            'entity_id'  => $entityId,
            'type'       => $tokenType,
            'created_at' => now(),
            'updated_at' => now(),
        ]);
    }
};