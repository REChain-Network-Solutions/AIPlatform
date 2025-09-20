# [YooKassa API SDK](../home.md)

# Interface: LegInterface
### Namespace: [\YooKassa\Request\Payments](../namespaces/yookassa-request-payments.md)
---
**Summary:**

Interface PassengerInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getDepartureAirport()](../classes/YooKassa-Request-Payments-LegInterface.md#method_getDepartureAirport) |  | Возвращает трёхбуквенный IATA-код аэропорта вылета. |
| public | [getDepartureDate()](../classes/YooKassa-Request-Payments-LegInterface.md#method_getDepartureDate) |  | Возвращает дату вылета в формате YYYY-MM-DD ISO 8601:2004. |
| public | [getDestinationAirport()](../classes/YooKassa-Request-Payments-LegInterface.md#method_getDestinationAirport) |  | Возвращает трёхбуквенный IATA-код аэропорта прилёта. |
| public | [setDepartureAirport()](../classes/YooKassa-Request-Payments-LegInterface.md#method_setDepartureAirport) |  | Устанавливает трёхбуквенный IATA-код аэропорта вылета. |
| public | [setDepartureDate()](../classes/YooKassa-Request-Payments-LegInterface.md#method_setDepartureDate) |  | Устанавливает дату вылета в формате YYYY-MM-DD ISO 8601:2004. |
| public | [setDestinationAirport()](../classes/YooKassa-Request-Payments-LegInterface.md#method_setDestinationAirport) |  | Устанавливает трёхбуквенный IATA-код аэропорта прилёта. |

---
### Details
* File: [lib/Request/Payments/LegInterface.php](../../lib/Request/Payments/LegInterface.php)
* Package: \YooKassa\Request
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Трёхбуквенный IATA-код аэропорта вылета |
| property |  | Трёхбуквенный IATA-код аэропорта вылета |
| property |  | Трёхбуквенный IATA-код аэропорта прилёта |
| property |  | Трёхбуквенный IATA-код аэропорта прилёта |
| property |  | Дата вылета в формате YYYY-MM-DD ISO 8601:2004 |
| property |  | Дата вылета в формате YYYY-MM-DD ISO 8601:2004 |
| property |  | Код авиакомпании по справочнику |
| property |  | Код авиакомпании по справочнику |

---
## Methods
<a name="method_getDepartureAirport" class="anchor"></a>
#### public getDepartureAirport() : string|null

```php
public getDepartureAirport() : string|null
```

**Summary**

Возвращает трёхбуквенный IATA-код аэропорта вылета.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\LegInterface](../classes/YooKassa-Request-Payments-LegInterface.md)

**Returns:** string|null - Трёхбуквенный IATA-код аэропорта вылета


<a name="method_getDestinationAirport" class="anchor"></a>
#### public getDestinationAirport() : string|null

```php
public getDestinationAirport() : string|null
```

**Summary**

Возвращает трёхбуквенный IATA-код аэропорта прилёта.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\LegInterface](../classes/YooKassa-Request-Payments-LegInterface.md)

**Returns:** string|null - Трёхбуквенный IATA-код аэропорта прилёта


<a name="method_getDepartureDate" class="anchor"></a>
#### public getDepartureDate() : \DateTime|null

```php
public getDepartureDate() : \DateTime|null
```

**Summary**

Возвращает дату вылета в формате YYYY-MM-DD ISO 8601:2004.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\LegInterface](../classes/YooKassa-Request-Payments-LegInterface.md)

**Returns:** \DateTime|null - Дата вылета в формате YYYY-MM-DD ISO 8601:2004


<a name="method_setDepartureAirport" class="anchor"></a>
#### public setDepartureAirport() : self

```php
public setDepartureAirport(string|null $value) : self
```

**Summary**

Устанавливает трёхбуквенный IATA-код аэропорта вылета.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\LegInterface](../classes/YooKassa-Request-Payments-LegInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Трёхбуквенный IATA-код аэропорта вылета |

**Returns:** self - 


<a name="method_setDestinationAirport" class="anchor"></a>
#### public setDestinationAirport() : self

```php
public setDestinationAirport(string|null $value) : self
```

**Summary**

Устанавливает трёхбуквенный IATA-код аэропорта прилёта.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\LegInterface](../classes/YooKassa-Request-Payments-LegInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Трёхбуквенный IATA-код аэропорта прилёта |

**Returns:** self - 


<a name="method_setDepartureDate" class="anchor"></a>
#### public setDepartureDate() : self

```php
public setDepartureDate(\DateTime|string|null $value) : self
```

**Summary**

Устанавливает дату вылета в формате YYYY-MM-DD ISO 8601:2004.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\LegInterface](../classes/YooKassa-Request-Payments-LegInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | value  | Дата вылета в формате YYYY-MM-DD ISO 8601:2004 |

**Returns:** self - 




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