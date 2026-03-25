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
            $table->json('payload_json')->nullable();
            $table->string('origin_node')->nullable();
            $table->string('team_id')->nullable();
            $table->string('status')->default('pending');
            $table->timestamp('created_at')->useCurrent();
            // queued_at intentionally tracks capture order, distinct from created_at audit stamp.
            $table->timestamp('processed_at')->nullable();
            $table->timestamp('queued_at')->useCurrent();
            $table->timestamp('updated_at')->nullable();
        });

        Schema::create('tz_sync_queue', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('object_type')->nullable();
            // object_id references the local signal id captured in tz_local_signals.
            $table->string('object_id')->nullable();
            $table->string('action')->nullable();
            $table->string('status')->default('pending');
            $table->unsignedInteger('retry_count')->default(0);
            $table->timestamp('last_attempt_at')->nullable();
            $table->json('payload_json')->nullable();
            $table->timestamp('created_at')->useCurrent();
            $table->timestamp('processed_at')->nullable();
            $table->timestamp('updated_at')->nullable();
        });

        Schema::create('tz_runtime_meta', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('subsystem')->nullable();
            $table->string('event')->nullable();
            $table->string('execution_layer')->nullable();
            $table->unsignedBigInteger('latency_ms')->nullable();
            $table->boolean('success_flag')->default(false);
            $table->json('metadata_json')->nullable();
            $table->timestamp('created_at')->useCurrent();
            $table->timestamp('updated_at')->nullable();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('tz_runtime_meta');
        Schema::dropIfExists('tz_sync_queue');
        Schema::dropIfExists('tz_local_signals');
    }
};
