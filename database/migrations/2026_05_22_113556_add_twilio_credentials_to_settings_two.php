<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('settings_two', function (Blueprint $table) {
            $table->string('twilio_account_sid')->nullable()->after('elevenlabs_api_key');
            $table->string('twilio_auth_token')->nullable()->after('twilio_account_sid');
        });
    }

    public function down(): void
    {
        Schema::table('settings_two', function (Blueprint $table) {
            $table->dropColumn(['twilio_account_sid', 'twilio_auth_token']);
        });
    }
};
