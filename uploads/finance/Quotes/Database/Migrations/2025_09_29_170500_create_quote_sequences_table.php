<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (!Schema::hasTable('quote_sequences')) {
            Schema::create('quote_sequences', function (Blueprint $table) {
                $table->bigIncrements('id');
                $table->string('series')->default('default');
                $table->integer('year')->nullable();
                $table->unsignedBigInteger('next_number')->default(1);
                $table->timestamps();
                $table->unique(['series','year']);
            });
        }
    }

    public function down(): void
    {
        Schema::dropIfExists('quote_sequences');
    }
};
