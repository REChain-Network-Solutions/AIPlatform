<?php

namespace YooKassa\Model\SavePaymentMethod;


use YooKassa\Model\Payment\PaymentMethod\BankCard;
use YooKassa\Model\SavePaymentMethod\Confirmation\AbstractConfirmation;

/**
 * Interface SavePaymentMethodInterface.
 *
 * Сохраненный способ оплаты.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property SavePaymentMethodType $type Код способа оплаты.  Возможное значение: ~`bank_card` — банковская карта.
 * @property string $id Идентификатор сохраненного способа оплаты.
 * @property bool $saved Признак сохранения способа оплаты для [автоплатежей](/developers/payment-acceptance/scenario-extensions/recurring-payments/pay-with-saved).  Возможные значения:   * ~`true` — способ оплаты сохранен для автоплатежей и выплат; * ~`false` — способ оплаты не сохранен.
 * @property SavePaymentMethodStatus $status Статус проверки и сохранения способа оплаты.
 * @property SavePaymentMethodHolder $holder Данные магазина, для которого сохраняется способ оплаты.
 * @property string $title Название способа оплаты.
 * @property AbstractConfirmation $confirmation Выбранный сценарий подтверждения привязки. Присутствует, когда привязка ожидает подтверждения от пользователя.
 */
interface SavePaymentMethodInterface
{
    /**
     * Возвращает код способа оплаты. Возможное значение: ~`bank_card` — банковская карта.
     *
     * @return string|null
     */
    public function getType(): ?string;

    /**
     * Возвращает card.
     *
     * @return BankCard|null
     */
    public function getCard(): ?BankCard;

    /**
     * Возвращает идентификатор сохраненного способа оплаты.
     *
     * @return string|null
     */
    public function getId(): ?string;

    /**
     * Возвращает признак сохранения способа оплаты для %[автоплатежей](/developers/payment-acceptance/scenario-extensions/recurring-payments/pay-with-saved).  Возможные значения:   * ~`true` — способ оплаты сохранен для автоплатежей и выплат; * ~`false` — способ оплаты не сохранен.
     *
     * @return bool|null
     */
    public function getSaved(): ?bool;

    /**
     * Возвращает статус проверки и сохранения способа оплаты.
     *
     * @return string|null
     */
    public function getStatus(): ?string;

    /**
     * Возвращает данные магазина, для которого сохраняется способ оплаты.
     *
     * @return SavePaymentMethodHolder|null
     */
    public function getHolder(): ?SavePaymentMethodHolder;

    /**
     * Возвращает название способа оплаты.
     *
     * @return string|null
     */
    public function getTitle(): ?string;

    /**
     * Возвращает выбранный сценарий подтверждения привязки.
     *
     * @return AbstractConfirmation|null
     */
    public function getConfirmation(): ?AbstractConfirmation;
}
