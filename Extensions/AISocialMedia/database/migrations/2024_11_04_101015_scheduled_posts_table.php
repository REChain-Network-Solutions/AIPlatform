<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasTable('scheduled_posts')) {
            return;
        }

        Schema::create('scheduled_posts', function (Blueprint $table) {
            $table->id();
            $table->bigInteger('user_id')->nullable();
            $table->bigInteger('company_id')->nullable();
            $table->string('platform')->nullable();
            $table->text('products')->nullable();
            $table->string('campaign_name')->nullable();
            $table->longText('campaign_target')->nullable();
            $table->longText('topics')->nullable();
            $table->boolean('is_seo')->default(0);
            $table->string('tone')->nullable();
            $table->string('length')->nullable();
            $table->boolean('is_email')->default(0);
            $table->boolean('is_repeated')->default(0);
            $table->string('repeat_period')->nullable();
            $table->date('repeat_start_date')->nullable();
            $table->time('repeat_time')->nullable();
            $table->string('visual_format')->nullable();
            $table->string('visual_ratio')->nullable();
            $table->string('posted_at')->nullable();
            $table->longtext('prompt')->nullable();
            $table->timestamps();
        });
    }

    public function down(): void {}
};
