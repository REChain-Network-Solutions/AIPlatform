# [YooKassa API SDK](../home.md)

# Interface: AirlineInterface
### Namespace: [\YooKassa\Request\Payments](../namespaces/yookassa-request-payments.md)
---
**Summary:**

Interface Airline.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getBookingReference()](../classes/YooKassa-Request-Payments-AirlineInterface.md#method_getBookingReference) |  | Номер бронирования. Обязателен на этапе создания платежа. |
| public | [getLegs()](../classes/YooKassa-Request-Payments-AirlineInterface.md#method_getLegs) |  | Список объектов-контейнеров с данными о маршруте. |
| public | [getPassengers()](../classes/YooKassa-Request-Payments-AirlineInterface.md#method_getPassengers) |  | Список объектов-контейнеров с данными пассажиров. |
| public | [getTicketNumber()](../classes/YooKassa-Request-Payments-AirlineInterface.md#method_getTicketNumber) |  | Уникальный номер билета. Обязателен на этапе подтверждения платежа. |

---
### Details
* File: [lib/Request/Payments/AirlineInterface.php](../../lib/Request/Payments/AirlineInterface.php)
* Package: \YooKassa\Request
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Номер бронирования. Обязателен на этапе создания платежа |
| property |  | Номер бронирования. Обязателен на этапе создания платежа |
| property |  | Уникальный номер билета. Обязателен на этапе подтверждения платежа |
| property |  | Уникальный номер билета. Обязателен на этапе подтверждения платежа |
| property |  | Список пассажиров |
| property |  | Список маршрутов |

---
## Methods
<a name="method_getBookingReference" class="anchor"></a>
#### public getBookingReference() : ?string

```php
public getBookingReference() : ?string
```

**Summary**

Номер бронирования. Обязателен на этапе создания платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AirlineInterface](../classes/YooKassa-Request-Payments-AirlineInterface.md)

**Returns:** ?string - 


<a name="method_getTicketNumber" class="anchor"></a>
#### public getTicketNumber() : ?string

```php
public getTicketNumber() : ?string
```

**Summary**

Уникальный номер билета. Обязателен на этапе подтверждения платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AirlineInterface](../classes/YooKassa-Request-Payments-AirlineInterface.md)

**Returns:** ?string - 


<a name="method_getPassengers" class="anchor"></a>
#### public getPassengers() : \YooKassa\Request\Payments\PassengerInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getPassengers() : \YooKassa\Request\Payments\PassengerInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Список объектов-контейнеров с данными пассажиров.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AirlineInterface](../classes/YooKassa-Request-Payments-AirlineInterface.md)

**Returns:** \YooKassa\Request\Payments\PassengerInterface[]|\YooKassa\Common\ListObjectInterface - 


<a name="method_getLegs" class="anchor"></a>
#### public getLegs() : \YooKassa\Request\Payments\LegInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getLegs() : \YooKassa\Request\Payments\LegInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Список объектов-контейнеров с данными о маршруте.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AirlineInterface](../classes/YooKassa-Request-Payments-AirlineInterface.md)

**Returns:** \YooKassa\Request\Payments\LegInterface[]|\YooKassa\Common\ListObjectInterface - 




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