# [YooKassa API SDK](../home.md)

# Interface: NotificationInterface
### Namespace: [\YooKassa\Model\Notification](../namespaces/yookassa-model-notification.md)
---
---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEvent()](../classes/YooKassa-Model-Notification-NotificationInterface.md#method_getEvent) |  | Возвращает тип события. |
| public | [getObject()](../classes/YooKassa-Model-Notification-NotificationInterface.md#method_getObject) |  | Возвращает объект с информацией о платеже или возврате, уведомление о котором хранится в текущем объекте. |
| public | [getType()](../classes/YooKassa-Model-Notification-NotificationInterface.md#method_getType) |  | Возвращает тип уведомления. |

---
### Details
* File: [lib/Model/Notification/NotificationInterface.php](../../lib/Model/Notification/NotificationInterface.php)
* Package: \Default

---
## Methods
<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип уведомления.

**Description**

Тип уведомления - одна из констант, указанных в перечислении {@link}.

**Details:**
* Inherited From: [\YooKassa\Model\Notification\NotificationInterface](../classes/YooKassa-Model-Notification-NotificationInterface.md)

**Returns:** string|null - Тип уведомления в виде строки


<a name="method_getEvent" class="anchor"></a>
#### public getEvent() : string|null

```php
public getEvent() : string|null
```

**Summary**

Возвращает тип события.

**Description**

Тип события - одна из констант, указанных в перечислении {@link}.

**Details:**
* Inherited From: [\YooKassa\Model\Notification\NotificationInterface](../classes/YooKassa-Model-Notification-NotificationInterface.md)

**Returns:** string|null - Тип события


<a name="method_getObject" class="anchor"></a>
#### public getObject() : \YooKassa\Model\Payment\PaymentInterface|\YooKassa\Model\Refund\RefundInterface|\YooKassa\Model\Payout\PayoutInterface|\YooKassa\Model\Deal\DealInterface|null

```php
public getObject() : \YooKassa\Model\Payment\PaymentInterface|\YooKassa\Model\Refund\RefundInterface|\YooKassa\Model\Payout\PayoutInterface|\YooKassa\Model\Deal\DealInterface|null
```

**Summary**

Возвращает объект с информацией о платеже или возврате, уведомление о котором хранится в текущем объекте.

**Description**

Так как нотификация может быть сгенерирована и поставлена в очередь на отправку гораздо раньше, чем она будет
получена на сайте, то опираться на статус пришедшего платежа не стоит, лучше запросить текущую информацию о
платеже у API.

**Details:**
* Inherited From: [\YooKassa\Model\Notification\NotificationInterface](../classes/YooKassa-Model-Notification-NotificationInterface.md)

**Returns:** \YooKassa\Model\Payment\PaymentInterface|\YooKassa\Model\Refund\RefundInterface|\YooKassa\Model\Payout\PayoutInterface|\YooKassa\Model\Deal\DealInterface|null - Объект с информацией о платеже




---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney