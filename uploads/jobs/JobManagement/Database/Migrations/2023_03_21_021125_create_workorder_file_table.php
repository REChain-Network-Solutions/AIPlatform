<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('workorders_files', function (Blueprint $table) {
            $table->increments('id');
            $table->unsignedInteger('company_id')->nullable();
            $table->foreign('company_id')->references('id')->on('companies')->onDelete('set null')->onUpdate('cascade');
            $table->unsignedInteger('user_id')->nullable();
            $table->foreign('user_id')->references('id')->on('users')->onDelete('cascade')->onUpdate('cascade');
            $table->unsignedInteger('workorder_id')->nullable();
            $table->foreign('workorder_id')->references('id')->on('workorders')->onDelete('cascade')->onUpdate('cascade');
            $table->string('filename');
            $table->text('description')->nullable()->default(null);
            $table->string('google_url')->nullable()->default(null);
            $table->string('hashname')->nullable()->default(null);
            $table->string('size')->nullable()->default(null);
            $table->string('dropbox_link')->nullable()->default(null);
            $table->string('external_link')->nullable()->default(null);
            $table->string('external_link_name')->nullable()->default(null);
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
        Schema::dropIfExists('workorders_files');
    }
};
