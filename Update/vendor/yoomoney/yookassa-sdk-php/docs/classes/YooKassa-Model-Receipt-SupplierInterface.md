# [YooKassa API SDK](../home.md)

# Interface: SupplierInterface
### Namespace: [\YooKassa\Model\Receipt](../namespaces/yookassa-model-receipt.md)
---
**Summary:**

Interface SupplierInterface.

**Description:**

Информация о поставщике товара или услуги.

Можно передавать, если вы отправляете данные для формирования чека по сценарию - сначала платеж, потом чек.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getInn()](../classes/YooKassa-Model-Receipt-SupplierInterface.md#method_getInn) |  | Возвращает ИНН пользователя (10 или 12 цифр). |
| public | [getName()](../classes/YooKassa-Model-Receipt-SupplierInterface.md#method_getName) |  | Возвращает наименование поставщика. |
| public | [getPhone()](../classes/YooKassa-Model-Receipt-SupplierInterface.md#method_getPhone) |  | Возвращает Телефон пользователя. Указывается в формате ITU-T E.164. |
| public | [setInn()](../classes/YooKassa-Model-Receipt-SupplierInterface.md#method_setInn) |  | Устанавливает ИНН пользователя (10 или 12 цифр). |
| public | [setName()](../classes/YooKassa-Model-Receipt-SupplierInterface.md#method_setName) |  | Устанавливает наименование поставщика. |
| public | [setPhone()](../classes/YooKassa-Model-Receipt-SupplierInterface.md#method_setPhone) |  | Устанавливает Телефон пользователя. Указывается в формате ITU-T E.164. |

---
### Details
* File: [lib/Model/Receipt/SupplierInterface.php](../../lib/Model/Receipt/SupplierInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |
| property |  | Наименование поставщика |
| property |  | Телефон пользователя. Указывается в формате ITU-T E.164 |
| property |  | ИНН пользователя (10 или 12 цифр) |

---
## Methods
<a name="method_getName" class="anchor"></a>
#### public getName() : ?string

```php
public getName() : ?string
```

**Summary**

Возвращает наименование поставщика.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\SupplierInterface](../classes/YooKassa-Model-Receipt-SupplierInterface.md)

**Returns:** ?string - 


<a name="method_setName" class="anchor"></a>
#### public setName() : mixed

```php
public setName(null|string $name) : mixed
```

**Summary**

Устанавливает наименование поставщика.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\SupplierInterface](../classes/YooKassa-Model-Receipt-SupplierInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | name  | Наименование поставщика |

**Returns:** mixed - 


<a name="method_getPhone" class="anchor"></a>
#### public getPhone() : null|string

```php
public getPhone() : null|string
```

**Summary**

Возвращает Телефон пользователя. Указывается в формате ITU-T E.164.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\SupplierInterface](../classes/YooKassa-Model-Receipt-SupplierInterface.md)

**Returns:** null|string - Телефон пользователя


<a name="method_setPhone" class="anchor"></a>
#### public setPhone() : mixed

```php
public setPhone(null|string $phone) : mixed
```

**Summary**

Устанавливает Телефон пользователя. Указывается в формате ITU-T E.164.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\SupplierInterface](../classes/YooKassa-Model-Receipt-SupplierInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | phone  | Телефон пользователя |

**Returns:** mixed - 


<a name="method_getInn" class="anchor"></a>
#### public getInn() : null|string

```php
public getInn() : null|string
```

**Summary**

Возвращает ИНН пользователя (10 или 12 цифр).

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\SupplierInterface](../classes/YooKassa-Model-Receipt-SupplierInterface.md)

**Returns:** null|string - ИНН пользователя


<a name="method_setInn" class="anchor"></a>
#### public setInn() : mixed

```php
public setInn(null|string $inn) : mixed
```

**Summary**

Устанавливает ИНН пользователя (10 или 12 цифр).

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\SupplierInterface](../classes/YooKassa-Model-Receipt-SupplierInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | inn  | ИНН пользователя |

**Returns:** mixed - 




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