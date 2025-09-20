# [YooKassa API SDK](../home.md)

# Interface: AuthorizationDetailsInterface
### Namespace: [\YooKassa\Model\Payment](../namespaces/yookassa-model-payment.md)
---
**Summary:**

Interface AuthorizationDetailsInterface - Данные об авторизации платежа.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAuthCode()](../classes/YooKassa-Model-Payment-AuthorizationDetailsInterface.md#method_getAuthCode) |  | Возвращает код авторизации банковской карты. |
| public | [getRrn()](../classes/YooKassa-Model-Payment-AuthorizationDetailsInterface.md#method_getRrn) |  | Возвращает Retrieval Reference Number — идентификатор банковской транзакции. |
| public | [getThreeDSecure()](../classes/YooKassa-Model-Payment-AuthorizationDetailsInterface.md#method_getThreeDSecure) |  | Возвращает данные о прохождении пользователем аутентификации по 3‑D Secure. |

---
### Details
* File: [lib/Model/Payment/AuthorizationDetailsInterface.php](../../lib/Model/Payment/AuthorizationDetailsInterface.php)
* Package: \Default

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| property |  | Retrieval Reference Number — идентификатор банковской транзакции |
| property |  | Код авторизации банковской карты |
| property |  | Код авторизации банковской карты |
| property |  | Данные о прохождении пользователем аутентификации по 3‑D Secure |
| property |  | Данные о прохождении пользователем аутентификации по 3‑D Secure |

---
## Methods
<a name="method_getRrn" class="anchor"></a>
#### public getRrn() : null|string

```php
public getRrn() : null|string
```

**Summary**

Возвращает Retrieval Reference Number — идентификатор банковской транзакции.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\AuthorizationDetailsInterface](../classes/YooKassa-Model-Payment-AuthorizationDetailsInterface.md)

**Returns:** null|string - Уникальный идентификатор транзакции


<a name="method_getAuthCode" class="anchor"></a>
#### public getAuthCode() : null|string

```php
public getAuthCode() : null|string
```

**Summary**

Возвращает код авторизации банковской карты.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\AuthorizationDetailsInterface](../classes/YooKassa-Model-Payment-AuthorizationDetailsInterface.md)

**Returns:** null|string - Код авторизации банковской карты


<a name="method_getThreeDSecure" class="anchor"></a>
#### public getThreeDSecure() : null|\YooKassa\Model\Payment\ThreeDSecure

```php
public getThreeDSecure() : null|\YooKassa\Model\Payment\ThreeDSecure
```

**Summary**

Возвращает данные о прохождении пользователем аутентификации по 3‑D Secure.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\AuthorizationDetailsInterface](../classes/YooKassa-Model-Payment-AuthorizationDetailsInterface.md)

**Returns:** null|\YooKassa\Model\Payment\ThreeDSecure - Объект с данными о прохождении пользователем аутентификации по 3‑D Secure




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