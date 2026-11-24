<?php

use App\Domains\Entity\Models\Entity;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    public function up(): void
    {
        Entity::query()
            ->where('engine', 'gemini')
            ->whereIn('key', [
                'gemini-3.5-flash',
                'gemini-3.5-flash-lite',
                'gemini-3.6-flash',
            ])
            ->update(['image' => 'upload/enginelogo/gemini_logo.svg']);

        Entity::query()
            ->where('engine', 'x_ai')
            ->whereIn('key', [
                'grok-4.5',
            ])
            ->update(['image' => 'upload/enginelogo/Grok_logo.svg']);
    }

    public function down(): void
    {
        Entity::query()
            ->where('engine', 'gemini')
            ->whereIn('key', [
                'gemini-3.5-flash',
                'gemini-3.5-flash-lite',
                'gemini-3.6-flash',
            ])
            ->where('image', 'upload/enginelogo/gemini_logo.svg')
            ->update(['image' => null]);

        Entity::query()
            ->where('engine', 'x_ai')
            ->whereIn('key', [
                'grok-4.5',
            ])
            ->where('image', 'upload/enginelogo/Grok_logo.svg')
            ->update(['image' => null]);
    }
};
