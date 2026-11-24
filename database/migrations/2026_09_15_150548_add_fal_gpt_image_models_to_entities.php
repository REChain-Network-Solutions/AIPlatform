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
     * @var array<int, array{key: string, title: string}>
     */
    private array $entities = [
        [
            'key'   => 'openai/gpt-image-2',
            'title' => 'GPT Image 2',
        ],
        [
            'key'   => 'openai/gpt-image-2.5/sunburst/text-to-image',
            'title' => 'GPT Image 2.5 Sunburst',
        ],
        [
            'key'   => 'openai/gpt-image-2.5/flare/text-to-image',
            'title' => 'GPT Image 2.5 Flare',
        ],
    ];

    public function up(): void
    {
        foreach ($this->entities as $entity) {
            $entityId = $this->upsertEntity($entity);

            $this->ensureImageToken($entityId);
        }

        Cache::forget('entities');
    }

    public function down(): void
    {
        if (Schema::hasTable('app_settings')) {
            DB::table('app_settings')
                ->where('key', 'fal_ai_default_model')
                ->whereIn('value', $this->storedModelKeys())
                ->update(['value' => 'flux-pro']);

            app('setting')->load(true);
        }

        DB::table('entities')
            ->whereIn('key', array_column($this->entities, 'key'))
            ->delete();

        Cache::forget('entities');
    }

    /**
     * @param  array{key: string, title: string}  $entity
     */
    private function upsertEntity(array $entity): int
    {
        $entityId = DB::table('entities')->where('key', $entity['key'])->value('id');
        $attributes = [
            'engine'     => 'fal_ai',
            'title'      => $entity['title'],
            'status'     => 'enabled',
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

    private function ensureImageToken(int $entityId): void
    {
        $tokenId = DB::table('tokens')
            ->where('entity_id', $entityId)
            ->where('type', 'image')
            ->value('id');

        if ($tokenId !== null) {
            DB::table('tokens')->where('id', $tokenId)->update(['updated_at' => now()]);

            return;
        }

        DB::table('tokens')->insert([
            'entity_id'  => $entityId,
            'type'       => 'image',
            'created_at' => now(),
            'updated_at' => now(),
        ]);
    }

    /**
     * @return array<int, string>
     */
    private function storedModelKeys(): array
    {
        return [
            'openai/gpt-image-2',
            'openai/gpt-image-2.5/sunburst/text-to-image',
            'openai/gpt-image-2.5/flare/text-to-image',
            'openai/gpt-image-2__5/sunburst/text-to-image',
            'openai/gpt-image-2__5/flare/text-to-image',
        ];
    }
};