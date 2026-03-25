<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void
    {
        Schema::create('tz_local_signals', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('signal_type')->nullable();
            $table->json('payload')->nullable();
            $table->string('status')->default('pending');
            $table->string('device_id')->nullable();
            $table->timestamp('queued_at')->useCurrent();
            $table->timestamp('dispatched_at')->nullable();
            $table->timestamps();
        });

        Schema::create('tz_sync_queue', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('direction')->default('outbound');
            $table->json('payload')->nullable();
            $table->string('status')->default('pending');
            $table->unsignedInteger('retry_count')->default(0);
            $table->text('last_error')->nullable();
            $table->string('device_id')->nullable();
            $table->timestamp('scheduled_at')->nullable();
            $table->timestamp('processed_at')->nullable();
            $table->timestamps();
        });

        Schema::create('tz_runtime_meta', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('category')->nullable();
            $table->string('meta_key')->nullable();
            $table->json('meta_value')->nullable();
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('tz_runtime_meta');
        Schema::dropIfExists('tz_sync_queue');
        Schema::dropIfExists('tz_local_signals');
    }
};
