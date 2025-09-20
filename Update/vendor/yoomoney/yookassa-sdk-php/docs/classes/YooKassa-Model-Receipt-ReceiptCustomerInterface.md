# [YooKassa API SDK](../home.md)

# Interface: ReceiptCustomerInterface
### Namespace: [\YooKassa\Model\Receipt](../namespaces/yookassa-model-receipt.md)
---
**Summary:**

Interface ReceiptCustomerInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEmail()](../classes/YooKassa-Model-Receipt-ReceiptCustomerInterface.md#method_getEmail) |  | Возвращает адрес электронной почты на который будет выслан чек. |
| public | [getFullName()](../classes/YooKassa-Model-Receipt-ReceiptCustomerInterface.md#method_getFullName) |  | Возвращает название организации или ФИО физического лица. |
| public | [getInn()](../classes/YooKassa-Model-Receipt-ReceiptCustomerInterface.md#method_getInn) |  | Возвращает ИНН плательщика. |
| public | [getPhone()](../classes/YooKassa-Model-Receipt-ReceiptCustomerInterface.md#method_getPhone) |  | Возвращает номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек. |
| public | [jsonSerialize()](../classes/YooKassa-Model-Receipt-ReceiptCustomerInterface.md#method_jsonSerialize) |  | Возвращает массив полей плательщика. |

---
### Details
* File: [lib/Model/Receipt/ReceiptCustomerInterface.php](../../lib/Model/Receipt/ReceiptCustomerInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |
| property |  | Для юрлица — название организации, для ИП и физического лица — ФИО. |
| property |  | Для юрлица — название организации, для ИП и физического лица — ФИО. |
| property |  | Номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек. |
| property |  | E-mail адрес плательщика на который будет выслан чек. |
| property |  | ИНН плательщика (10 или 12 цифр). |

---
## Methods
<a name="method_getFullName" class="anchor"></a>
#### public getFullName() : string|null

```php
public getFullName() : string|null
```

**Summary**

Возвращает название организации или ФИО физического лица.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomerInterface](../classes/YooKassa-Model-Receipt-ReceiptCustomerInterface.md)

**Returns:** string|null - Название организации или ФИО физического лица


<a name="method_getPhone" class="anchor"></a>
#### public getPhone() : string|null

```php
public getPhone() : string|null
```

**Summary**

Возвращает номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomerInterface](../classes/YooKassa-Model-Receipt-ReceiptCustomerInterface.md)

**Returns:** string|null - Номер телефона плательщика


<a name="method_getEmail" class="anchor"></a>
#### public getEmail() : string|null

```php
public getEmail() : string|null
```

**Summary**

Возвращает адрес электронной почты на который будет выслан чек.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomerInterface](../classes/YooKassa-Model-Receipt-ReceiptCustomerInterface.md)

**Returns:** string|null - E-mail адрес плательщика


<a name="method_getInn" class="anchor"></a>
#### public getInn() : string|null

```php
public getInn() : string|null
```

**Summary**

Возвращает ИНН плательщика.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomerInterface](../classes/YooKassa-Model-Receipt-ReceiptCustomerInterface.md)

**Returns:** string|null - ИНН плательщика


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает массив полей плательщика.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomerInterface](../classes/YooKassa-Model-Receipt-ReceiptCustomerInterface.md)

**Returns:** array - 




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