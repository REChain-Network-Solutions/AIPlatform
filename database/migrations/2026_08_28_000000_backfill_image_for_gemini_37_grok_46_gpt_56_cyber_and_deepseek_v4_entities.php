<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

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
            'key'        => 'gemini-3.7-flash',
            'title'      => 'Gemini 3.7 Flash Frontier multimodal intelligence with a 1M token context window, built for agentic and high-throughput workloads',
            'engine'     => 'gemini',
            'token_type' => 'word',
            'logo'       => 'upload/enginelogo/gemini_logo.svg',
        ],
        [
            'key'        => 'grok-4.6',
            'title'      => 'Grok 4.6 (Jan 2026 knowledge cutoff, 500k context. Pricing doubles for prompts of 200k tokens or more.)',
            'engine'     => 'x_ai',
            'token_type' => 'word',
            'logo'       => 'upload/enginelogo/Grok_logo.svg',
        ],
        [
            'key'        => 'gpt-5.6-cyber',
            'title'      => 'GPT-5.6 Cyber (Security research model, 128k max output tokens, 400k context. Requires separate OpenAI approval via the Daybreak program.)',
            'engine'     => 'openai',
            'token_type' => 'word',
            'logo'       => 'upload/enginelogo/openai.svg',
        ],
        [
            'key'        => 'deepseek-v4-flash',
            'title'      => 'DeepSeek V4 Flash Fast, low-cost model with a 1M token context window for high-throughput tasks',
            'engine'     => 'deep_seek',
            'token_type' => 'word',
            'logo'       => 'upload/enginelogo/deepseek_logo.svg',
        ],
        [
            'key'        => 'deepseek-v4-pro',
            'title'      => 'DeepSeek V4 Pro Flagship reasoning model with a 1M token context window',
            'engine'     => 'deep_seek',
            'token_type' => 'word',
            'logo'       => 'upload/enginelogo/deepseek_logo.svg',
        ],
        [
            'key'        => 'deepseek-v4-flash-vision-exp',
            'title'      => 'DeepSeek V4 Flash Vision Experimental multimodal model with image understanding',
            'engine'     => 'deep_seek',
            'token_type' => 'word',
            'logo'       => 'upload/enginelogo/deepseek_logo.svg',
        ],
    ];

    /**
     * @var array<int, string>
     */
    private array $retiredDeepseekModels = ['deepseek-chat', 'deepseek-reasoner'];

    public function up(): void
    {
        foreach ($this->entities as $entity) {
            $entityId = $this->upsertEntity($entity);

            $this->ensureToken($entityId, $entity['token_type']);
        }

        $this->repointRetiredDeepseekDefault('deepseek-v4-pro');
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

    private function repointRetiredDeepseekDefault(string $replacement): void
    {
        if (! Schema::hasTable('app_settings')) {
            return;
        }

        DB::table('app_settings')
            ->where('key', 'deepseek_default_model')
            ->whereIn('value', $this->retiredDeepseekModels)
            ->update(['value' => $replacement]);

        app('setting')->load(true);
    }
};