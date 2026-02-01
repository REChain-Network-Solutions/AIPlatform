<?php

use App\Extensions\Chatbot\System\Models\ChatbotAvatar;
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\Storage;

return new class extends Migration
{
    public static $prefix = 'ext';

    public function up(): void
    {
        if (Schema::hasTable(self::$prefix . '_chatbot_avatars')) {
            return;
        }

        Schema::create(self::$prefix . '_chatbot_avatars', function (Blueprint $table) {
            $table->id();
            $table->bigInteger('user_id')->nullable();
            $table->string('avatar');
            $table->timestamp('created_at')->useCurrent();
        });

        $avatars = config('chatbot.avatars');

        if (ChatbotAvatar::query()->count() > 0) {
            return;
        }

        try {
            foreach ($avatars as $avatar) {

                $image = Storage::disk('extension')->path('Chatbot/resources/assets/avatars/' . $avatar);

                if (\Illuminate\Support\Facades\File::exists($image) === false) {
                    continue;
                }

                $file = Storage::disk('public')->putFile('avatars', $image);

                ChatbotAvatar::query()->create([
                    'avatar'     => 'uploads/' . $file,
                    'created_at' => now(),
                ]);
            }
        } catch (\Exception $e) {

        }

    }

    public function down(): void
    {
        Schema::dropIfExists(self::$prefix . '_chatbot_avatars');
    }
};
