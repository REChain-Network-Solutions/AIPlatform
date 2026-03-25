<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateBugFilesTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        if(!Schema::hasTable('bug_files'))
        {
            Schema::create('bug_files', function (Blueprint $table) {
                $table->id();
                $table->string('file');
                $table->string('name');
                $table->string('extension');
                $table->string('file_size');
                $table->integer('bug_id');
                $table->string('user_type');
                $table->integer('created_by');
                $table->timestamps();
            });
        }
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('bug_files');
    }
}
