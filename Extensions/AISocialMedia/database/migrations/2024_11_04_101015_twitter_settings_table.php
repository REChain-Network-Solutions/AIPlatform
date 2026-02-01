<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasTable('twitter_settings')) {
            return;
        }

        Schema::create('twitter_settings', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('user_id')->nullable();
            $table->text('consumer_key')->nullable();
            $table->text('consumer_secret')->nullable();
            $table->text('access_token')->nullable();
            $table->text('access_token_secret')->nullable();
            $table->text('bearer_token')->nullable();
            $table->text('account_id')->nullable();
            $table->timestamps();
        });
    }

    public function down(): void {}
};
