# CustomerFeedback Module - Installation Guide

## Prerequisites

- Laravel 9.0+
- PHP 8.1+
- MySQL 5.7+ or PostgreSQL 10+
- Composer

## Installation Steps

### 1. Extract Module

```bash
# Copy the CustomerFeedback folder to your Modules directory
cp -r CustomerFeedback /path/to/your/project/Modules/
```

### 2. Enable Module

Add to `modules_statuses.json`:

```json
{
    "CustomerFeedback": 1
}
```

### 3. Register Service Provider

If using manual provider registration, add to `config/app.php`:

```php
'providers' => [
    // ...
    Modules\CustomerFeedback\Providers\CustomerFeedbackServiceProvider::class,
    Modules\CustomerFeedback\Providers\EventServiceProvider::class,
],
```

### 4. Dump Autoloader

```bash
composer dump-autoload
```

### 5. Run Migrations

```bash
php artisan migrate
```

This will create:
- `feedback_tickets` - Main ticket table
- `feedback_replies` - Reply thread table
- `feedback_channels` - Channel definitions
- `feedback_types` - Type classifications
- `feedback_groups` - Agent groups
- `feedback_agent_groups` - Agent assignments
- `feedback_files` - Attachments
- `feedback_tags_list` & `feedback_tags` - Tagging system
- `feedback_reply_templates` - Response templates
- `feedback_custom_forms` - Dynamic forms
- `feedback_email_settings` - Email configuration
- `nps_surveys` & `nps_responses` - NPS surveys
- `csat_surveys` & `csat_responses` - CSAT surveys
- `feedback_insights` - AI analysis results

### 6. Publish Assets (Optional)

```bash
# Publish migrations
php artisan vendor:publish --tag=customer-feedback-migrations

# Publish config
php artisan vendor:publish --tag=customer-feedback-config

# Publish views
php artisan vendor:publish --tag=customer-feedback-views

# Publish translations
php artisan vendor:publish --tag=customer-feedback-lang
```

### 7. Create Initial Data

Run seeders to set up default channels, types, and groups:

```bash
php artisan db:seed --class=Modules\\CustomerFeedback\\Database\\Seeders\\CustomerFeedbackDatabaseSeeder
```

## Configuration

Edit `config/customer-feedback.php`:

```php
return [
    'module_settings' => [
        'enable_nps' => true,
        'enable_csat' => true,
        'enable_ai_insights' => true,
        'enable_email_sync' => true,
    ],

    'ai' => [
        'enabled' => true,
        'min_confidence' => 0.7,
    ],
];
```

## Usage

### Access the Module

Navigate to:
- Admin Dashboard: `http://yourapp.com/customer-feedback/tickets`
- Analytics: `http://yourapp.com/customer-feedback/analytics/dashboard`
- NPS Surveys: `http://yourapp.com/customer-feedback/surveys/nps`
- Insights: `http://yourapp.com/customer-feedback/insights/dashboard`

### Legacy Complaint Routes

Old complaint routes still work:
- `http://yourapp.com/complaint` → redirects to `/customer-feedback`

## Troubleshooting

### Migration Errors

If you get foreign key errors:

```bash
# Disable foreign key checks temporarily
php artisan tinker
>>> DB::statement('SET FOREIGN_KEY_CHECKS=0;')
>>> exit()

# Then run migrations
php artisan migrate

# Re-enable foreign key checks
php artisan tinker
>>> DB::statement('SET FOREIGN_KEY_CHECKS=1;')
>>> exit()
```

### Permission Denied

Ensure your web server user has write permissions:

```bash
sudo chown -R www-data:www-data /path/to/project
sudo chmod -R 755 /path/to/project
```

### Queue Jobs Not Processing

If AI analysis isn't working, ensure queue is running:

```bash
php artisan queue:work
# Or in production:
php artisan queue:work --daemon
```

## Post-Installation

1. **Create Channels**
   - Go to Settings and create feedback channels
   - Default: Email, Web Form, API, SMS, Chat

2. **Create Types**
   - Create complaint, feedback, and survey types
   - Set type categories appropriately

3. **Create Agent Groups**
   - Create groups and assign agents
   - Configure group permissions

4. **Configure Email**
   - If email sync needed, add IMAP credentials
   - Enable auto-reply if desired

5. **Create Surveys**
   - Create NPS and CSAT surveys
   - Configure survey questions

6. **Set Permissions**
   - Grant users appropriate feedback module permissions
   - Configure role-based access control

## Next Steps

- Read [README.md](README.md) for feature documentation
- Check [API.md](API.md) for API endpoints
- Review [CONFIGURATION.md](CONFIGURATION.md) for advanced settings

## Support

For issues or questions:
- Check error logs: `storage/logs/laravel.log`
- Verify database: `php artisan tinker` → `DB::table('feedback_tickets')->count()`
- Test permissions: `User::first()->permission('view_feedback')`

---

**Installation Complete!** 🎉

You can now start using the CustomerFeedback module.
