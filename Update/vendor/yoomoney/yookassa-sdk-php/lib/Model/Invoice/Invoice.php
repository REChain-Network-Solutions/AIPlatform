<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Model\Invoice;

use DateTime;
use YooKassa\Common\AbstractObject;
use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod;
use YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodFactory;
use YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodSelf;
use YooKassa\Model\Metadata;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель Invoice.
 *
 * Объект счета (Invoice) — актуальная информация о счете.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $id Идентификатор счета в ЮKassa.
 * @property string $status Статус счета. Возможные значения: `pending`, `succeeded`, `canceled`.
 * @property LineItem[]|ListObjectInterface $cart Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой.
 * @property AbstractDeliveryMethod|null $delivery_method Данные о выбранном способе доставки счета. Присутствует только для счетов в статусе `pending`.
 * @property AbstractDeliveryMethod|null $deliveryMethod Данные о выбранном способе доставки счета. Присутствует только для счетов в статусе `pending`.
 * @property PaymentDetails|null $payment_details Данные о платеже по выставленному счету. Присутствуют, только если платеж успешно [подтвержден пользователем](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#user-confirmation).
 * @property PaymentDetails|null $paymentDetails Данные о платеже по выставленному счету. Присутствуют, только если платеж успешно [подтвержден пользователем](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#user-confirmation).
 * @property DateTime $created_at Дата и время создания счета на оплату. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2017-11-03T11:52:31.827Z`
 * @property DateTime $createdAt Дата и время создания счета на оплату. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2017-11-03T11:52:31.827Z`
 * @property DateTime|null $expires_at Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2024-10-18T10:51:18.139Z` Присутствует только для счетов в статусе `pending`.
 * @property DateTime|null $expiresAt Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2024-10-18T10:51:18.139Z` Присутствует только для счетов в статусе `pending`.
 * @property string|null $description Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37».
 * @property InvoiceCancellationDetails|null $cancellation_details Комментарий к статусу `canceled`: кто отменил счет и по какой причине.
 * @property InvoiceCancellationDetails|null $cancellationDetails Комментарий к статусу `canceled`: кто отменил счет и по какой причине.
 * @property Metadata|null $metadata Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
 */
class Invoice extends AbstractObject implements InvoiceInterface
{
    /**
     * Идентификатор счета в ЮKassa.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_ID)]
    #[Assert\Length(min: self::MIN_LENGTH_ID)]
    protected ?string $_id = null;

    /**
     * Статус счета. Возможные значения: `pending`, `succeeded`, `canceled`.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [InvoiceStatus::class, 'getValidValues'])]
    #[Assert\Type('string')]
    protected ?string $_status = null;

    /**
     * Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой.
     *
     * @var LineItem[]|ListObjectInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\AllType(LineItem::class)]
    #[Assert\Type(ListObject::class)]
    protected ?ListObjectInterface $_cart = null;

    /**
     * Данные о выбранном способе доставки счета. Присутствует только для счетов в статусе `pending`.
     *
     * @var AbstractDeliveryMethod|null
     */
    #[Assert\Type(AbstractDeliveryMethod::class)]
    protected ?AbstractDeliveryMethod $_delivery_method = null;

    /**
     * Данные о платеже по выставленному счету. Присутствуют, только если платеж успешно [подтвержден пользователем](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#user-confirmation).
     *
     * @var PaymentDetails|null
     */
    #[Assert\Type(PaymentDetails::class)]
    protected ?PaymentDetails $_payment_details = null;

    /**
     * Дата и время создания счета на оплату. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
     * Пример: ~`2017-11-03T11:52:31.827Z`
     *
     * @var DateTime|null
     */
    #[Assert\NotBlank]
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type(DateTime::class)]
    protected ?DateTime $_created_at = null;

    /**
     * Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
     * Пример: `2024-10-18T10:51:18.139Z`
     *
     * Присутствует только для счетов в статусе `pending`.
     *
     * @var DateTime|null
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type(DateTime::class)]
    protected ?DateTime $_expires_at = null;

    /**
     * Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета.
     * Например: «Счет на оплату по договору 37».
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_DESCRIPTION)]
    protected ?string $_description = null;

    /**
     * Комментарий к статусу `canceled`: кто отменил счет и по какой причине.
     *
     * @var InvoiceCancellationDetails|null
     */
    #[Assert\Valid]
    #[Assert\Type(InvoiceCancellationDetails::class)]
    protected ?InvoiceCancellationDetails $_cancellation_details = null;

    /**
     * Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).
     * Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa.
     * Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
     *
     * @var Metadata|null
     */
    #[Assert\Type(Metadata::class)]
    protected ?Metadata $_metadata = null;

    /**
     * Возвращает id.
     *
     * @return string|null Идентификатор счета в ЮKassa
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает id.
     *
     * @param string|null $id Идентификатор счета в ЮKassa.
     *
     * @return self
     */
    public function setId(?string $id = null): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }

    /**
     * Возвращает status.
     *
     * @return string|null Статус счета
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Устанавливает status.
     *
     * @param string|null $status Статус счета
     *
     * @return self
     */
    public function setStatus(?string $status = null): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * Возвращает cart.
     *
     * @return LineItem[]|ListObjectInterface|null Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой
     */
    public function getCart(): ?ListObjectInterface
    {
        if ($this->_cart === null) {
            $this->_cart = new ListObject(LineItem::class);
        }
        return $this->_cart;
    }

    /**
     * Устанавливает cart.
     *
     * @param ListObjectInterface|LineItem[]|array|null $cart Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой
     *
     * @return self
     */
    public function setCart(mixed $cart = null): self
    {
        $this->_cart = $this->validatePropertyValue('_cart', $cart);
        return $this;
    }

    /**
     * Возвращает delivery_method.
     *
     * @return AbstractDeliveryMethod|DeliveryMethodSelf|null Данные о выбранном способе доставки счета
     */
    public function getDeliveryMethod(): ?AbstractDeliveryMethod
    {
        return $this->_delivery_method;
    }

    /**
     * Устанавливает delivery_method.
     *
     * @param AbstractDeliveryMethod|array|null $delivery_method Данные о выбранном способе доставки счета
     *
     * @return self
     */
    public function setDeliveryMethod(mixed $delivery_method = null): self
    {
        if (is_array($delivery_method)) {
            $delivery_method = (new DeliveryMethodFactory())->factoryFromArray($delivery_method);
        }
        $this->_delivery_method = $this->validatePropertyValue('_delivery_method', $delivery_method);
        return $this;
    }

    /**
     * Возвращает payment_details.
     *
     * @return PaymentDetails|null Данные о платеже по выставленному счету
     */
    public function getPaymentDetails(): ?PaymentDetails
    {
        return $this->_payment_details;
    }

    /**
     * Устанавливает payment_details.
     *
     * @param PaymentDetails|array|null $payment_details Данные о платеже по выставленному счету
     *
     * @return self
     */
    public function setPaymentDetails(mixed $payment_details = null): self
    {
        $this->_payment_details = $this->validatePropertyValue('_payment_details', $payment_details);
        return $this;
    }

    /**
     * Возвращает created_at.
     *
     * @return DateTime|null Дата и время создания счета на оплату
     */
    public function getCreatedAt(): ?DateTime
    {
        return $this->_created_at;
    }

    /**
     * Устанавливает created_at.
     *
     * @param DateTime|string|null $created_at Дата и время создания счета на оплату
     *
     * @return self
     */
    public function setCreatedAt(DateTime|string|null $created_at = null): self
    {
        $this->_created_at = $this->validatePropertyValue('_created_at', $created_at);
        return $this;
    }

    /**
     * Возвращает expires_at.
     *
     * @return DateTime|null Срок действия счета
     */
    public function getExpiresAt(): ?DateTime
    {
        return $this->_expires_at;
    }

    /**
     * Устанавливает expires_at.
     *
     * @param DateTime|array|null $expires_at Срок действия счета
     *
     * @return self
     */
    public function setExpiresAt(mixed $expires_at = null): self
    {
        $this->_expires_at = $this->validatePropertyValue('_expires_at', $expires_at);
        return $this;
    }

    /**
     * Возвращает description.
     *
     * @return string|null Описание выставленного счета
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Устанавливает description.
     *
     * @param string|null $description Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37».
     *
     * @return self
     */
    public function setDescription(?string $description = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * Возвращает cancellation_details.
     *
     * @return InvoiceCancellationDetails|null Комментарий к статусу `canceled`
     */
    public function getCancellationDetails(): ?InvoiceCancellationDetails
    {
        return $this->_cancellation_details;
    }

    /**
     * Устанавливает cancellation_details.
     *
     * @param InvoiceCancellationDetails|array|null $cancellation_details Комментарий к статусу `canceled`
     *
     * @return self
     */
    public function setCancellationDetails(mixed $cancellation_details = null): self
    {
        $this->_cancellation_details = $this->validatePropertyValue('_cancellation_details', $cancellation_details);
        return $this;
    }

    /**
     * Возвращает metadata.
     *
     * @return Metadata|null Любые дополнительные данные
     */
    public function getMetadata(): ?Metadata
    {
        return $this->_metadata;
    }

    /**
     * Устанавливает metadata.
     *
     * @param Metadata|array|null $metadata Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).
     *
     * @return self
     */
    public function setMetadata(mixed $metadata = null): self
    {
        $this->_metadata = $this->validatePropertyValue('_metadata', $metadata);
        return $this;
    }

}

