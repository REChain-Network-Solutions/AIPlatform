<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\DB;

return new class extends Migration
{
    private const KEY = 'claude-fable-5-1';

    private const LOGO = 'upload/enginelogo/claude_logo.svg';

    public function up(): void
    {
        $entityId = DB::table('entities')->where('key', self::KEY)->value('id');
        $attributes = [
            'engine'     => 'anthropic',
            'title'      => 'Claude Fable 5.1',
            'status'     => 'enabled',
            'image'      => self::LOGO,
            'updated_at' => now(),
        ];

        if ($entityId !== null) {
            DB::table('entities')->where('id', $entityId)->update($attributes);
        } else {
            $entityId = DB::table('entities')->insertGetId([
                'key'        => self::KEY,
                'created_at' => now(),
                ...$attributes,
            ]);
        }

        $this->ensureWordToken((int) $entityId);
        Cache::forget('entities');
    }

    public function down(): void
    {
        DB::table('entities')
            ->where('key', self::KEY)
            ->where('image', self::LOGO)
            ->update([
                'image'      => null,
                'updated_at' => now(),
            ]);

        Cache::forget('entities');
    }

    private function ensureWordToken(int $entityId): void
    {
        $tokenId = DB::table('tokens')
            ->where('entity_id', $entityId)
            ->where('type', 'word')
            ->value('id');

        if ($tokenId !== null) {
            DB::table('tokens')->where('id', $tokenId)->update(['updated_at' => now()]);

            return;
        }

        DB::table('tokens')->insert([
            'entity_id'  => $entityId,
            'type'       => 'word',
            'created_at' => now(),
            'updated_at' => now(),
        ]);
    }
};