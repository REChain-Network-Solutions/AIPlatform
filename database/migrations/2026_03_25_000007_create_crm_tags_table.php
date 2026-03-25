<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('crm_tags', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->cascadeOnDelete();
            $table->string('name');
            $table->string('color', 7)->default('#6366f1');
            $table->timestamps();

            $table->unique(['user_id', 'name']);
        });

        Schema::create('crm_taggables', function (Blueprint $table) {
            $table->foreignId('crm_tag_id')->constrained('crm_tags')->cascadeOnDelete();
            $table->morphs('taggable'); // taggable_id, taggable_type
            $table->primary(['crm_tag_id', 'taggable_id', 'taggable_type']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('crm_taggables');
        Schema::dropIfExists('crm_tags');
    }
};
