# AuditLog Module

A comprehensive audit logging system for tracking all database changes across the ERP system.

## Features

- **Automatic Tracking**: Tracks create, update, delete, and restore events for all models
- **User Attribution**: Records which user made each change
- **Technical Details**: Captures IP address, user agent, and URL
- **Change Comparison**: Shows old vs new values for all modified fields
- **Advanced Filtering**: Filter by model, event type, user, and date range
- **Visual Analytics**: Charts showing daily trends and most active users
- **Conditional Logging**: Only logs when the module is enabled

## Installation

The module is automatically installed as part of the ERP system. To enable/disable:

1. Go to Settings > Addons
2. Find "AuditLog" in the list
3. Toggle the status

## Usage

### For Models

To enable audit logging for a model, use the `ConditionalAuditable` trait:

```php
use Modules\AuditLog\app\Traits\ConditionalAuditable;

class YourModel extends Model
{
    use ConditionalAuditable;
    
    // Your model code
}
```

### Accessing Audit Logs

Navigate to the Audit Logs section in the main menu to:
- View all audit logs with advanced filtering
- See detailed changes for each record
- View analytics and trends

## Configuration

The module respects the main audit configuration in `config/audit.php`. When the AuditLog module is disabled, no audit events will be recorded.

## API Endpoints

- `GET /auditlog` - Main audit logs page
- `GET /auditlog/datatable` - DataTable AJAX endpoint
- `GET /auditlog/statistics` - Statistics data
- `GET /auditlog/filters` - Available filter options
- `GET /auditlog/{id}` - View specific audit log details

## Permissions

Access to audit logs is controlled by the existing permission system. Users need appropriate permissions to view audit logs.