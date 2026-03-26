<?php

defined('BASEPATH') or exit('No direct script access allowed');

if (!$CI->db->table_exists(db_prefix() . 'rb_settings')) {
    $CI->db->query('CREATE TABLE `' . db_prefix() . "rb_settings` (
      `id` INT(11) UNSIGNED NOT NULL AUTO_INCREMENT,
      `report_title` TEXT NULL,
      `category_id` INT(11) NULL,
      `report_footer` TEXT NULL,
      `report_header` TEXT NULL,
      `report_name` VARCHAR(200) NULL,
      `records_per_page` INT(11) NULL,
      `layout_name` VARCHAR(100) NULL,
      `style_name` VARCHAR(100) NULL,
      `is_public` VARCHAR(100) NULL DEFAULT 'no',
      `role_id` TEXT NULL,
      `department_id` TEXT NULL,
      `staff_id` TEXT NULL,

      PRIMARY KEY (`id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=" . $CI->db->char_set . ';');
}

if (!$CI->db->table_exists(db_prefix() . 'rb_templates')) {
    $CI->db->query('CREATE TABLE `' . db_prefix() . "rb_templates` (
      `id` INT(11) UNSIGNED NOT NULL AUTO_INCREMENT,
      `report_title` TEXT NULL,
      `category_id` INT(11) NULL,
      `report_footer` TEXT NULL,
      `report_header` TEXT NULL,
      `report_name` VARCHAR(200) NULL,
      `records_per_page` INT(11) NULL,
      `layout_name` VARCHAR(100) NULL,
      `style_name` VARCHAR(100) NULL,
      `is_public` VARCHAR(100) NULL DEFAULT 'no',
      `role_id` TEXT NULL,
      `department_id` TEXT NULL,
      `staff_id` TEXT NULL,
      `staff_create` INT(11) NULL,
      `date_create` DATETIME NULL,

      `group_by_field` TEXT NULL,

      PRIMARY KEY (`id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=" . $CI->db->char_set . ';');
}


if (!$CI->db->table_exists(db_prefix() . 'rb_data_source_relationships')) {
    $CI->db->query('CREATE TABLE `' . db_prefix() . "rb_data_source_relationships` (
      `id` INT(11) UNSIGNED NOT NULL AUTO_INCREMENT,
      `templates_id` INT(11) NULL,
      `left_table` VARCHAR(200) NULL,
      `left_field_1` VARCHAR(200) NULL,
      `left_field_2` VARCHAR(200) NULL,
      `right_table` VARCHAR(200) NULL,
      `right_field_1` VARCHAR(200) NULL,
      `right_field_2` VARCHAR(200) NULL,
      `query_string` TEXT NULL,

      PRIMARY KEY (`id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=" . $CI->db->char_set . ';');
}

if (!$CI->db->table_exists(db_prefix() . 'rb_data_source_filters')) {
    $CI->db->query('CREATE TABLE `' . db_prefix() . "rb_data_source_filters` (
      `id` INT(11) UNSIGNED NOT NULL AUTO_INCREMENT,
      `templates_id` INT(11) NULL,
      `table_name` VARCHAR(200) NULL,
      `field_name` VARCHAR(200) NULL,
      `filter_type` VARCHAR(200) NULL,
      `filter_value_1` TEXT NULL,
      `filter_value_2` TEXT NULL,
      `query_filter` TEXT NULL,

      PRIMARY KEY (`id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=" . $CI->db->char_set . ';');
}

if (!$CI->db->table_exists(db_prefix() . 'rb_columns')) {
    $CI->db->query('CREATE TABLE `' . db_prefix() . "rb_columns` (
      `id` INT(11) UNSIGNED NOT NULL AUTO_INCREMENT,
      `templates_id` INT(11) NULL,
      `table_name` VARCHAR(200) NULL,
      `field_name` VARCHAR(200) NULL,
      `label_name` VARCHAR(200) NULL,
      `field_type` VARCHAR(200) NULL,
      `group_by` VARCHAR(200) NULL DEFAULT 'no',
      `function_name` VARCHAR(200) NULL,
      `allow_subtotal` VARCHAR(200) NULL DEFAULT 'no',
      `affected_column` VARCHAR(200) NULL DEFAULT 'no',

      PRIMARY KEY (`id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=" . $CI->db->char_set . ';');
}

if (!$CI->db->table_exists(db_prefix() . 'rb_aggregation_functions')) {
    $CI->db->query('CREATE TABLE `' . db_prefix() . "rb_aggregation_functions` (
      `id` INT(11) UNSIGNED NOT NULL AUTO_INCREMENT,
      `templates_id` INT(11) NULL,
      `statistical_function` VARCHAR(200) NULL,
      `affected_column` VARCHAR(200) NULL,
      `group_by` VARCHAR(200) NULL,
      `allow_aggregation_function` VARCHAR(200) NULL DEFAULT 'no',

      PRIMARY KEY (`id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=" . $CI->db->char_set . ';');
}

if (!$CI->db->table_exists(db_prefix() . 'rb_field_conditional_formattings')) {
    $CI->db->query('CREATE TABLE `' . db_prefix() . "rb_field_conditional_formattings` (
      `id` INT(11) UNSIGNED NOT NULL AUTO_INCREMENT,
      `templates_id` INT(11) NULL,
      `table_name` VARCHAR(200) NULL,
      `field_name` VARCHAR(200) NULL,
      `filter_type` VARCHAR(200) NULL,
      `filter_value_1` TEXT NULL,
      `filter_value_2` TEXT NULL,
      `color_hex` VARCHAR(100) NULL,
      `query_filter` TEXT NULL,

      PRIMARY KEY (`id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=" . $CI->db->char_set . ';');
}

if (!$CI->db->table_exists(db_prefix() . 'rb_categories')) {
	$CI->db->query('CREATE TABLE `' . db_prefix() . "rb_categories` (
		`id` int(11) UNSIGNED NOT NULL AUTO_INCREMENT,
		`name` text NULL ,

		PRIMARY KEY (`id`)
	) ENGINE=InnoDB DEFAULT CHARSET=" . $CI->db->char_set . ';');
}

//order by
if (!$CI->db->table_exists(db_prefix() . 'rb_sort_bys')) {
    $CI->db->query('CREATE TABLE `' . db_prefix() . "rb_sort_bys` (
      `id` INT(11) UNSIGNED NOT NULL AUTO_INCREMENT,
      `templates_id` INT(11) NULL,
      `table_name` VARCHAR(200) NULL,
      `field_name` VARCHAR(200) NULL,
      `order_by` VARCHAR(200) NULL DEFAULT 'ASC',

      PRIMARY KEY (`id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=" . $CI->db->char_set . ';');
}

if (!$CI->db->field_exists('join_column' ,db_prefix() . 'rb_data_source_relationships')) { 
  $CI->db->query('ALTER TABLE `' . db_prefix() . "rb_data_source_relationships`

    ADD COLUMN `join_column` VARCHAR(200) NULL,
    ADD COLUMN `join_type` VARCHAR(100) NULL

  ;");
}

if (!$CI->db->field_exists('ask_user' ,db_prefix() . 'rb_data_source_filters')) { 
  $CI->db->query('ALTER TABLE `' . db_prefix() . "rb_data_source_filters`

    ADD COLUMN `ask_user` VARCHAR(200) NULL,
    ADD COLUMN `group_condition` VARCHAR(100) NULL

  ;");
}

if (!$CI->db->field_exists('order_display' ,db_prefix() . 'rb_columns')) { 
  $CI->db->query('ALTER TABLE `' . db_prefix() . "rb_columns`

    ADD COLUMN `order_display` INT(200) NULL

  ;");
}

if (!$CI->db->field_exists('except_staff' ,db_prefix() . 'rb_templates')) { 
  $CI->db->query('ALTER TABLE `' . db_prefix() . "rb_templates`

    ADD COLUMN `except_staff` TEXT NULL

  ;");
}
if (!$CI->db->field_exists('rel_type_condition' ,db_prefix() . 'rb_data_source_relationships')) { 
  $CI->db->query('ALTER TABLE `' . db_prefix() . "rb_data_source_relationships`

    ADD COLUMN `rel_type_condition` TEXT NULL

  ;");
}

