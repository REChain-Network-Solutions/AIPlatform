# Shared Credit Pool - Roadmap

Mevcut "separated credits" (model bazli ayri ayri kredi) sisteminin yanina paralel calisan "shared credit pool" (tek havuz) sistemi. Kullanici tek bir bakiyeden tum AI modellerini kullanir; her modelin farkli maliyet carpani (`sharedCreditIndex`) vardir.

## Mimari

```
Entity::driver($entity)->forUser($user)
    -> HasCreditLimit::getCreditIndex()
        -> isSharedCreditUser()?
            -> EVET: sharedCreditIndex() veya DB override
            -> HAYIR: creditIndex() (1.0, degismez)
    -> calculate(): quantity * getCreditIndex()
    -> decreaseCredit()
        -> EVET: SharedCreditService::deduct() (shared_credits)
        -> HAYIR: entity_credits JSON (mevcut)
```

## Fazlar

| Faz | Baslik | Durum | Bagimlilik | Kapsam |
|-----|--------|-------|------------|--------|
| [Faz 1](faz1.md) | Backend Altyapisi | TAMAMLANDI | - | DB, Model, Service, Trait, Test |
| [Faz 2](faz2.md) | Admin Panel - Plan Yonetimi | Bekliyor | Faz 1 | Livewire plan form, badge, assign-view-credits |
| [Faz 3](faz3.md) | Admin Panel - Maliyet & Kullanici Yonetimi | Bekliyor | Faz 1 | Cost CRUD, settings, user finance, transaction log |
| [Faz 4](faz4.md) | Kullanici Dashboard Frontend | Bekliyor | Faz 2 | credit-list, gecmis, breakdown, maliyet onizleme |
| [Faz 5](faz5.md) | Team Shared Credits | Bekliyor | Faz 4 | Team havuz, member dusme, gunluk limit |
| [Faz 6](faz6.md) | Gelismis Ozellikler | Bekliyor | Faz 4 | Expiration, auto top-up, limits, notifications, API, packages |

## Bagimlilik Grafi

```
Faz 1 (TAMAMLANDI)
  |
  +-- Faz 2 (Plan Yonetimi)
  |     |
  |     +-- Faz 4 (Kullanici Frontend)
  |           |
  |           +-- Faz 5 (Team)
  |           |
  |           +-- Faz 6 (Gelismis)
  |                 6.1 Expiration
  |                 6.2 Auto Top-up
  |                 6.3 Feature Limits
  |                 6.4 Notifications
  |                 6.5 API
  |                 6.6 Top-up Packages
  |
  +-- Faz 3 (Maliyet & Kullanici) [Faz 2 ile paralel]
```

Faz 2 ve 3 paralel gelistirilebilir. Faz 4 icin Faz 2 gerekli. Faz 5 ve 6 alt adimlari bagimsiz eklenebilir.

## Referans Dokumanlar

| Dokuman | Aciklama |
|---------|----------|
| [shared-credit-pool-v2.md](../shared-credit-pool-v2.md) | Detayli teknik plan (8 bolum, tum analiz) |
| [shared-credit-pool-teknik-plan.md](../shared-credit-pool-teknik-plan.md) | Ilk taslak teknik plan |
