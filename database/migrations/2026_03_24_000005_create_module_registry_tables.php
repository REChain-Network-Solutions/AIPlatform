<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void
    {
        if (!Schema::hasTable('module_registry_feeds')) {
            Schema::create('module_registry_feeds', function (Blueprint $table) {
                $table->id();
                $table->string('name');
                $table->string('slug')->unique();
                $table->string('source_type')->default('file');
                $table->text('source_url')->nullable();
                $table->json('auth_config')->nullable();
                $table->boolean('is_active')->default(true);
                $table->unsignedInteger('priority')->default(100);
                $table->timestamp('last_synced_at')->nullable();
                $table->text('last_error')->nullable();
                $table->json('sync_meta')->nullable();
                $table->timestamps();
            });
        }

        if (!Schema::hasTable('module_registry_items')) {
            Schema::create('module_registry_items', function (Blueprint $table) {
                $table->id();
                $table->foreignId('feed_id')->nullable()->constrained('module_registry_feeds')->nullOnDelete();
                $table->string('module_name');
                $table->string('slug');
                $table->string('latest_version')->nullable();
                $table->json('versions')->nullable();
                $table->json('compatible_with')->nullable();
                $table->json('dependencies')->nullable();
                $table->json('distribution')->nullable();
                $table->json('signature')->nullable();
                $table->string('status')->default('active');
                $table->timestamp('last_seen_at')->nullable();
                $table->json('meta')->nullable();
                $table->timestamps();
                $table->unique(['feed_id', 'slug']);
                $table->index(['module_name', 'latest_version']);
                $table->index('status');
            });
        }
    }

    public function down(): void
    {
        Schema::dropIfExists('module_registry_items');
        Schema::dropIfExists('module_registry_feeds');
    }
};
