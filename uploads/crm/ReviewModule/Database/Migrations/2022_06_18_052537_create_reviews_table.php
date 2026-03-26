<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateReviewsTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('reviews', function (Blueprint $table) {
            $table->uuid('id')->primary()->index();
            $table->unsignedBigInteger('company_id')->nullable()->index();
            $table->foreignUuid('booking_id')->nullable();
            $table->foreignUuid('service_id')->nullable();
            $table->foreignUuid('provider_id')->nullable();
            $table->integer('review_rating')->default(1);
            $table->text('review_comment')->nullable();
            $table->text('review_images')->nullable();
            $table->dateTime('booking_date')->nullable();
            $table->boolean('is_active')->default(1);
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('reviews');
    }
}
